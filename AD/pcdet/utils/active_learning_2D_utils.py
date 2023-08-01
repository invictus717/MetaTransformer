import enum
import io
import os
import tqdm
import pickle
import random
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances



def active_evaluate(model, target_loader, rank):
    if rank == 0:
        print("======> Active Evaluate <======")
    dataloader_iter_tar = iter(target_loader)
    total_iter_tar = len(dataloader_iter_tar)
    frame_scores = []
    return_scores = []
    model.eval()
    if rank == 0:
        pbar = tqdm.tqdm(total=total_iter_tar, leave=False, desc='active_evaluate', dynamic_ncols=True)

    for cur_it in range(total_iter_tar):
        try:
            batch = next(dataloader_iter_tar)
        except StopIteration:
            dataloader_iter_tar = iter(target_loader)
            batch = next(dataloader_iter_tar)
            print('new iter')

        with torch.no_grad():
            load_data_to_gpu(batch)
            forward_args = {
                'mode': 'active_evaluate'
            }
            sample_score = model(batch, **forward_args)
            frame_scores.append(sample_score)

        if rank == 0:
            pbar.update()
            pbar.refresh()
    
    if rank == 0:
        pbar.close()

    gather_scores = gather_all_scores(frame_scores)
    for score in gather_scores:
        for f_score in score:
            return_scores += f_score
    return return_scores


def active_evaluate_dual(model, target_loader, rank, domain):
    if rank == 0:
        print("======> Active Evaluate <======")
    dataloader_iter_tar = iter(target_loader)
    total_iter_tar = len(dataloader_iter_tar)
    frame_scores = []
    return_scores = []
    model.eval()
    if rank == 0:
        pbar = tqdm.tqdm(total=total_iter_tar, leave=False, desc='active_evaluate', dynamic_ncols=True)

    for cur_it in range(total_iter_tar):
        try:
            batch = next(dataloader_iter_tar)
        except StopIteration:
            dataloader_iter_tar = iter(target_loader)
            batch = next(dataloader_iter_tar)
            print('new iter')

        with torch.no_grad():
            load_data_to_gpu(batch)
            forward_args = {
                'mode': 'active_evaluate',
                'domain': domain
            }
            sample_score = model(batch, **forward_args)
            frame_scores.append(sample_score)

        if rank == 0:
            pbar.update()
            pbar.refresh()
    
    if rank == 0:
        pbar.close()

    gather_scores = gather_all_scores(frame_scores)
    for score in gather_scores:
        for f_score in score:
            return_scores += f_score
    
    return return_scores

# evaluate all frame (including sampled frame)
def active_evaluate_dual_2(model, target_loader, rank, domain, sampled_frame_id=None):
    if rank == 0:
        print("======> Active Evaluate <======")
    dataloader_iter_tar = iter(target_loader)
    total_iter_tar = len(dataloader_iter_tar)
    frame_scores = []
    sampled_frame_scores = []
    return_scores = []
    model.eval()
    if rank == 0:
        pbar = tqdm.tqdm(total=total_iter_tar, leave=False, desc='active_evaluate', dynamic_ncols=True)

    for cur_it in range(total_iter_tar):
        try:
            batch = next(dataloader_iter_tar)
        except StopIteration:
            dataloader_iter_tar = iter(target_loader)
            batch = next(dataloader_iter_tar)
            print('new iter')

        with torch.no_grad():
            load_data_to_gpu(batch)
            forward_args = {
                'mode': 'active_evaluate',
                'domain': domain
            }
            sample_score = model(batch, **forward_args)
            frame_scores.append(sample_score)

        if rank == 0:
            pbar.update()
            pbar.refresh()
    
    if rank == 0:
        pbar.close()

    gather_scores = gather_all_scores(frame_scores)
    for score in gather_scores:
        for f_score in score:
            return_scores += f_score
    
    for i, score in enumerate(return_scores):
        if score['frame_id'] in sampled_frame_id:
            sampled_frame_scores.append(score)
            return_scores.pop(i)
    return return_scores, sampled_frame_scores

def gather_all_scores(frame_scores):
    commu_utils.synchronize()
    if dist.is_initialized():
        scores = commu_utils.all_gather(frame_scores)
    else:
        scores = [frame_scores]
    commu_utils.synchronize()
    return scores

def distributed_concat(tensor):
    output_tensor = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensor, tensor)
    concat_tensor = torch.cat(output_tensor, dim=0)
    return concat_tensor 

def get_target_list(target_pkl_file, oss):
    if oss == True:
        from petrel_client.client import Client
        client = Client('~/.petreloss.conf')
        pkl_bytes = client.get(target_pkl_file, update_cache=True)
        target_list = pickle.load(io.BytesIO(pkl_bytes))
    else:
        with open(target_pkl_file, 'rb') as f:
            target_list = pickle.load(f)
    return target_list



def get_dataset_list(dataset_file, oss, sample_interval=10, waymo=False):
    if oss == True:
        from petrel_client.client import Client
        client = Client('~/.petreloss.conf')
    if waymo == False:
        if oss == True:
            # from petrel_client.client import Client
            # client = Client('~/.petreloss.conf')
            pkl_bytes = client.get(dataset_file, update_cache=True)
            target_list = pickle.load(io.BytesIO(pkl_bytes))
        else:
            with open(dataset_file, 'rb') as f:
                target_list = pickle.load(f)
    else:
        data_path = '../data/waymo/ImageSets/train.txt'
        target_list = []
        sample_sequence_list = [x.strip() for x in open(data_path).readlines()]
        for k in tqdm.tqdm(range(len(sample_sequence_list))):
            sequence_name = os.path.splitext(sample_sequence_list[k])[0]
            if oss == False:
                info_path = Path(dataset_file) / sequence_name / ('%s.pkl' % sequence_name)
                if not Path(info_path).exists():
                    continue
            else:
                info_path = os.path.join(dataset_file, sequence_name, ('%s.pkl' % sequence_name))
                # if not Path(info_path).exists():
                #     continue
            
            if oss == False:
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    target_list.extend(infos)
            else:
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                target_list.extend(infos)
        if sample_interval > 1:
            sampled_waymo_infos = []
            for k in range(0, len(target_list), sample_interval):
                sampled_waymo_infos.append(target_list[k])
            target_list = sampled_waymo_infos
    return target_list



def update_sample_list(sample_list, target_list, sample_frame_id, epoch, save_path, target_name, rank):
    if target_name == 'ActiveKittiDataset':
        new_sample_list = [item for item in target_list if item['point_cloud']['lidar_idx'] in sample_frame_id]
    elif target_name == 'ActiveNuScenesDataset':
        new_sample_list = [item for item in target_list if Path(item['lidar_path']).stem in sample_frame_id]
    sample_list = sample_list + new_sample_list
    sample_list_path = save_path / ('epoch-%d_sample_list.pkl' % epoch)
    if rank == 0:
        with open(sample_list_path, 'wb') as f:
            pickle.dump(sample_list, f)
    commu_utils.synchronize()
    return sample_list, sample_list_path


def update_sample_list_dual(sample_list, dataset_list, sample_frame_id, epoch, save_path, dataset_name, rank, domain='source'):
    if dataset_name == 'ActiveKittiDataset':
        assert domain == 'target'
        new_sample_list = [item for item in dataset_list if item['point_cloud']['lidar_idx'] in sample_frame_id]
        sample_list = sample_list + new_sample_list
    elif dataset_name == 'ActiveNuScenesDataset':
        if domain == 'target':
            new_sample_list = [item for item in dataset_list if Path(item['lidar_path']).stem in sample_frame_id]
            sample_list = sample_list + new_sample_list
        else:
            sample_list = [item for item in dataset_list if Path(item['lidar_path']).stem in sample_frame_id]
    elif dataset_name == 'ActiveWaymoDataset':
        assert domain == 'source'
        # if rank == 0:
        #     print(sample_frame_id)
        sample_list = [item for item in dataset_list if str(item['frame_id']) in sample_frame_id]

        # if rank == 0:
        #     print('dataset_list: %d' % len(dataset_list))
        #     print('sample frame number: %d' % len(sample_list))
    sample_list_path = save_path / ('epoch-%d_sample_list_'  % epoch + '_' + domain + '.pkl')

    if rank == 0:
        with open(sample_list_path, 'wb') as f:
            pickle.dump(sample_list, f)
    commu_utils.synchronize()
    return sample_list, sample_list_path


def update_target_list(target_list, sample_frame_id, epoch, save_path, target_name, rank):
    if target_name == 'ActiveKittiDataset':
        target_list = [item for item in target_list if item['point_cloud']['lidar_idx'] not in sample_frame_id]
    elif target_name == 'ActiveNuScenesDataset':
        target_list = [item for item in target_list if Path(item['lidar_path']).stem not in sample_frame_id]
    target_list_path = save_path / ('epoch-%d_target_list.pkl' % epoch)
    if rank == 0:
        with open(target_list_path, 'wb') as f:
            pickle.dump(target_list, f)
    commu_utils.synchronize()
    return target_list, target_list_path


def active_sample(frame_scores, budget):
    frame_sorted = sorted(frame_scores, key=lambda keys: keys.get("total_score"), reverse=True)
    sampled_frame_info = frame_sorted[:budget]
    sampled_frame_id = [frame['frame_id'] for frame in sampled_frame_info]
    # fused_frame_info = [item for item in frame_scores if item['total_score'] > 0]
    # print('fused_frame: %d' % len(fused_frame_info))
    return sampled_frame_id, sampled_frame_info


def active_sample_source(frame_scores, budget):
    sampled_frame_info = [item for item in frame_scores if item['total_score'] > 0]
    sampled_frame_id = [frame['frame_id'] for frame in sampled_frame_info]
    return sampled_frame_id, sampled_frame_info

def active_sample_CLUE(frame_scores, budget):
    roi_feature = frame_scores[0].get('roi_feature', None)
    tgt_emb_pen = roi_feature.new_zeros((len(frame_scores), roi_feature.shape[-1]))
    tgt_scores = roi_feature.new_zeros((len(frame_scores), 1))
    for i, cur_score in enumerate (frame_scores):
        cur_feature = cur_score.get('roi_feature', None).to(tgt_emb_pen.device)
        cur_roi_score = cur_score.get('roi_score', None).to(tgt_scores.device)
        tgt_emb_pen[i] = cur_feature
        tgt_scores[i] = cur_roi_score
    tgt_emb_pen = tgt_emb_pen.cpu().numpy()
    tgt_scores = tgt_scores.view(-1)
    sample_weights = -(tgt_scores*torch.log(tgt_scores)).cpu().numpy()
    km = KMeans(budget)
    km.fit(tgt_emb_pen, sample_weight=sample_weights)

    dists = euclidean_distances(km.cluster_centers_, tgt_emb_pen)
    sort_idxs = dists.argsort(axis=1)
    q_idxs = []
    ax, rem = 0, budget
    while rem > 0:
        q_idxs.extend(list(sort_idxs[:, ax][:rem]))
        q_idxs = list(set(q_idxs))
        rem = budget - len(q_idxs)
        ax += 1
    sample_frame_info = []
    sample_frame_id = []
    for i, cur_score in enumerate(frame_scores):
        if i in q_idxs:
            sample_frame_info.append(cur_score)
            sample_frame_id.append(cur_score.get('frame_id'))
    return sample_frame_id, sample_frame_info
    
