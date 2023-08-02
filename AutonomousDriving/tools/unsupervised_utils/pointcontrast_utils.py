import os
import glob
# from plotly import data
from pcdet.models import load_data_to_gpu
import torch
import tqdm
from pcdet.models import load_data_to_gpu
from torch.nn.utils import clip_grad_norm_
from ssl_utils.semi_utils import random_world_flip, random_world_rotation, random_world_scaling
from pcdet.models.detectors.unsupervised_model.pvrcnn_plus_backbone import HardestContrastiveLoss

# @torch.no_grad()
# def get_positive_pairs(batch_dict_1, batch_dict_2):
#     augmentation_functions = {
#         'random_world_flip': random_world_flip,
#         'random_world_rotation': random_world_rotation,
#         'random_world_scaling': random_world_scaling
#     }
#     for bs_idx in range(len(batch_dict_1)):
#         aug_list_1 = batch_dict_1['augmentation_list'][bs_idx]
#         aug_list_2 = batch_dict_2['augmentation_list'][bs_idx]
#         aug_param_1 = batch_dict_1['augmentation_params'][bs_idx]
#         aug_param_2 = batch_dict_2['augmentation_params'][bs_idx]
        

def pointcontrast(model, batch_dict_1, batch_dict_2, loss_cfg, dist, voxel_size, point_cloud_range):
    
    load_data_to_gpu(batch_dict_1)
    load_data_to_gpu(batch_dict_2)

    if not dist:
        batch_dict_1 = model(batch_dict_1)
        batch_dict_2 = model(batch_dict_2)
    else:
        batch_dict_1, batch_dict_2 = model(batch_dict_1, batch_dict_2)
    
    contrastive_loss = HardestContrastiveLoss(loss_cfg, voxel_size, point_cloud_range)
    pos_loss, neg_loss = contrastive_loss.get_hardest_contrastive_loss(batch_dict_1, batch_dict_2)

    loss = pos_loss + neg_loss

    return loss


def train_pointcontrast_one_epoch(model, optimizer, data_loader, lr_scheduler, 
                                  voxel_size, point_cloud_range,
                                  accumulated_iter, cfg, rank, tbar, total_it_each_epoch, 
                                  dataloader_iter, tb_log=None, leave_pbar=False, dist=False):
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    disp_dict = {}
    for cur_epoch in range(total_it_each_epoch):
        try:
            batch_1, batch_2 = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch_1, batch_2 = next(dataloader_iter)
            print('new sample dataloader')
        
        try:
            cur_lr = float(optimizer.lr)
        except StopIteration:
            cur_lr = optimizer.param_group[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        
        optimizer.zero_grad()

        loss = pointcontrast(model, batch_1, batch_2, cfg.LOSS_CFG, dist, voxel_size, point_cloud_range)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), cfg.GRAD_NORM_CLIP)
        optimizer.step()
        lr_scheduler.step(accumulated_iter)

        accumulated_iter += 1
        disp_dict.update({
            'loss': loss.item(),
            'lr': cur_lr
        })

        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                # for key, val in tb_dict.items():
                #     tb_log.add_scalar('train/' + key, val, accumulated_iter)
    
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader, lr_scheduler, cfg, voxel_size, point_cloud_range,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                train_sampler, lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, dist=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader) # total iterations set to labeled set
        assert merge_all_iters_to_one_epoch is False
        train_loader_iter = iter(train_loader)

        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_pointcontrast_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, 
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size, cfg=cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dist = dist,
                dataloader_iter=train_loader_iter
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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 2), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        """
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        else:
            ema_param.data.mul_(0).add_(1, param.data)
        """


def update_ema_variables_with_fixed_momentum(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        """
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        else:
            ema_param.data.mul_(0).add_(1, param.data)
        """