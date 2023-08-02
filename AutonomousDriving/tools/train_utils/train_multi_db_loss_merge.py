import torch
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_


def visualize_boxes_batch(batch):
    import visualize_utils as vis
    import mayavi.mlab as mlab
    for b_idx in range(batch['batch_size']):
        points = batch['points'][batch['points'][:, 0] == b_idx][:, 1:]

        if 'debug' not in batch:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            scores=batch['scores'][b_idx])
        else:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            gt_boxes=batch['debug'][b_idx]['gt_boxes_lidar'],
                            scores=batch['scores'][b_idx])
        mlab.show(stop=True)

def merge_two_batch_data(batch_1, batch_2):
    import numpy as np
    ret = {}
    for key, val in batch_1.items():
        if key in ['batch_size']:
            continue
        else:
            ret[key] = np.stack(val, axis=0)
    for key, val in batch_2.items():
        val_cat = []
        if key in ['batch_size']:
            continue
        elif key in ['gt_boxes']:
            assert batch_1[key][0].shape[-1] == val[0].shape[-1]
            max_gt = max([len(x) for x in batch_1[key]]) + max([len(x) for x in val])
            batch_gt_boxes3d = np.zeros((batch_1['batch_size']*2, max_gt, val[0].shape[-1]), dtype=np.float32)
            for k in range(batch_1['batch_size']):
                batch_gt_boxes3d[k, :batch_1[key][k].__len__(), :] = batch_1[key][k]
            for k in range(batch_2['batch_size']):
                batch_gt_boxes3d[k+batch_1['batch_size'], :val[k].__len__(), :] = val[k]
            ret[key] = batch_gt_boxes3d
        else:
            val_cat.append(batch_1[key])
            val_cat.append(val)
            ret[key] = np.concatenate(val_cat, axis=0)
            #ret[key] = np.stack(val, axis=0)
    ret['batch_size'] = batch_1['batch_size']*2
    return ret

def train_one_epoch_multi_db(model, optimizer, train_loader_1, train_loader_2, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter_1, dataloader_iter_2, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader_1):
        dataloader_iter_1 = iter(train_loader_1)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        # Load the source domain ONE:
        try:
            batch_1 = next(dataloader_iter_1)
        except StopIteration:
            dataloader_iter_1 = iter(train_loader_1)
            batch_1 = next(dataloader_iter_1)
        # Load the source domain TWO:
        try:
            batch_2 = next(dataloader_iter_2)
        except StopIteration:
            dataloader_iter_2 = iter(train_loader_2)
            batch_2 = next(dataloader_iter_2)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        #  The loss_1 + loss_2 will lead to a runtime error of loss.backward() 
        #  when perofrming the pytorch distributed
        #  you should perform the forward and backward one by one.

        #  Loss for source domain ONE:
        loss_s1, tb_dict_s1, disp_dict_s1 = model_func(model, batch_1)
        #  Loss for source domain TWO:
        loss_s2, tb_dict_s2, _ = model_func(model, batch_2)
        #  Merge the two loss
        loss = loss_s1 + optim_cfg.DB_2_W * loss_s2

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict_s1.update({'loss': loss_s1.item(), 'lr': cur_lr})

        # log to console and tensorboard
        # save the log of the source domain ONE
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict_s1)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss_1', loss_s1, accumulated_iter)
                tb_log.add_scalar('train/loss_2', loss_s2, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict_s1.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                for key, val in tb_dict_s2.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_multi_db_model(model, optimizer, train_src_loader, train_src_loader_2, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_src_loader) if len(train_src_loader) > len(train_src_loader_2) else len(train_src_loader_2)

        if merge_all_iters_to_one_epoch:
            raise NotImplementedError

        dataloader_iter_1 = iter(train_src_loader)
        dataloader_iter_2 = iter(train_src_loader_2)
        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch_multi_db(
                model, optimizer, train_src_loader, train_src_loader_2, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter_1=dataloader_iter_1,
                dataloader_iter_2=dataloader_iter_2
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)