import io
import os
import tqdm
import pickle
import random
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils



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
    elif dataset_name =='ActiveLyftDataset':
        assert domain == 'target'
        new_sample_list = [item for item in dataset_list if Path(item['lidar_path']).stem in sample_frame_id]
        sample_list = sample_list + new_sample_list
        
    elif dataset_name == 'ActiveWaymoDataset':
        assert domain == 'source'
        sample_list = [item for item in dataset_list if str(item['frame_id']) in sample_frame_id]

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
    return sampled_frame_id, sampled_frame_info


def active_sample_source(frame_scores, budget):
    sampled_frame_info = [item for item in frame_scores if item['total_score'] > 0]
    sampled_frame_id = [frame['frame_id'] for frame in sampled_frame_info]
    return sampled_frame_id, sampled_frame_info


def active_sample_tar(frame_scores, budget, logger=None):
    roi_feature = frame_scores[0].get('roi_feature', None)
    feature_dim = roi_feature.shape[-1]
    sample_roi_feature = roi_feature.new_zeros((budget, feature_dim))
    prototype_roi_feature = roi_feature.new_zeros((budget, feature_dim))
    roi_feature = roi_feature.new_zeros((budget+1, feature_dim))
    domainness_list = []
    feature_list = []
    sample_feature_list = []
    for item in frame_scores:
        cur_roi_feature = item.get('roi_feature')
        cur_roi_feature = cur_roi_feature.to(roi_feature.device)
        cur_domainness = item.get('domainness_evaluate')
        cur_domainness = cur_domainness.to(roi_feature.device)
        if len(feature_list) < budget:
            sample_roi_feature[len(feature_list)] = cur_roi_feature
            prototype_roi_feature[len(feature_list)] = cur_roi_feature
            roi_feature[len(feature_list)] = cur_roi_feature

            feature_list.append([item])
            sample_feature_list.append(item)
            domainness_list.append(cur_domainness)
        else:
            roi_feature[-1, :] = cur_roi_feature
            similarity_matrix = F.normalize(roi_feature, dim=1) @ F.normalize(roi_feature, dim=1).transpose(1,0).contiguous()
            similarity, inds = similarity_matrix.topk(k=2, dim=0)
            similarity = similarity[1]
            inds = inds[1]
            if similarity.min(dim=-1)[0] == similarity[-1]:
                similarity_max = similarity.max(dim=-1)
                inds_y = similarity.argmax(dim=-1)
                inds_x = inds[inds_y]
                domainness_x = domainness_list[inds_x]
                domainness_y = domainness_list[inds_y]
                roi_feature_1 = sample_roi_feature[inds_x]
                roi_feature_2 = sample_roi_feature[inds_y]
                num_merge_prototype_1 = len(feature_list[inds_x])
                num_merge_prototype_2 = len(feature_list[inds_y])
                merge_proto = (num_merge_prototype_1 * prototype_roi_feature[inds_x] + num_merge_prototype_2 * prototype_roi_feature[inds_y]) / (num_merge_prototype_1 + num_merge_prototype_2)
                if domainness_x > domainness_y:
                    prototype_roi_feature[inds_x] = merge_proto
                    sample_roi_feature[inds_y] = cur_roi_feature
                    roi_feature[inds_y] = cur_roi_feature
                    feature_list[inds_x] = feature_list[inds_x] + feature_list[inds_y]
                    feature_list[inds_y] = [item]
                    sample_feature_list[inds_y] = item
                    domainness_list[inds_y] = cur_domainness
                else:
                    prototype_roi_feature[inds_y] = merge_proto
                    sample_roi_feature[inds_x] = cur_roi_feature
                    roi_feature[inds_x] = cur_roi_feature
                    feature_list[inds_y] = feature_list[inds_x] + feature_list[inds_y]
                    feature_list[inds_x] = [item]
                    sample_feature_list[inds_x] = item
                    domainness_list[inds_x] = cur_domainness

            else:
                merge_inds = inds[budget]
                merge_domainness = domainness_list[merge_inds]
                num_merge_proto = len(feature_list[merge_inds])
                merge_proto = (num_merge_proto * prototype_roi_feature[merge_inds] + cur_roi_feature) / (num_merge_proto + 1)
                prototype_roi_feature[merge_inds] = merge_proto
                if cur_domainness > merge_domainness:
                    sample_roi_feature[merge_inds] = cur_roi_feature
                    roi_feature[merge_inds] = cur_roi_feature
                    sample_feature_list[merge_inds] = item
                    domainness_list[merge_inds] = cur_domainness
                feature_list[merge_inds].append(item)
    for l in feature_list:
        sorted(l, key=lambda keys: keys.get("total_score"), reverse=True)
    
    num_each_group = [len(l) for l in feature_list]
    distance_matrix = F.normalize(prototype_roi_feature, dim=1) @ F.normalize(prototype_roi_feature, dim=1).transpose(0, 1).contiguous()
    distance, inds = distance_matrix.topk(k=2, dim=0)
    distance_each_group = distance[1]
    sample_num_each_group = distance_each_group.cpu() * torch.Tensor(num_each_group).cpu()
    sample_num_each_group = assign_sample_num(sample_num_each_group.numpy().tolist(), budget)
    sample_feature_list_new = []
    for i, sample_num in enumerate(sample_num_each_group):
        sampled = 0
        while sampled < sample_num:
            sample_feature_list_new.append(feature_list[i][sampled])
            sampled += 1
    sample_list_new = []
    for item in sample_feature_list_new:
        sample_list_new.append(item['frame_id'])
    
    sample_list = []
    for i, list in enumerate(feature_list):
        num = len(list)
        print('group %d has %d sample_frame' % (i, sample_num_each_group[i]))
        print('group %d has %d frame' % (i, num))
        if logger != None:
            logger.info('group %d has %d sample_frame' % (i, sample_num_each_group[i]))
            logger.info('group %d has %d frame' % (i, num))
    for item in sample_feature_list:
        sample_list.append(item['frame_id'])
    

    return sample_list_new, sample_feature_list_new


def assign_sample_num(sample_num_list, budget):
    sampled_num = 0
    sample_num_each_list = [0] * len(sample_num_list)
    while sampled_num < budget:
        ind = sample_num_list.index(max(sample_num_list))
        sample_num_each_list[ind] += 1
        sample_num_list[ind] /= 2
        sampled_num += 1
    return sample_num_each_list


def random_sample(source_list, target_list, source_budget, target_budget, save_path):
    random.shuffle(target_list)
    random.shuffle(source_list)
    target_sample_list = random.sample(target_list, target_budget)
    source_sample_list = random.sample(source_list, source_budget)
    target_sample_path = save_path / ('random_target_list.pkl')
    source_sample_path = save_path / ('random_source_list.pkl')
    with open(target_sample_path, 'wb') as f:
        pickle.dump(target_sample_list, f)
    with open(source_sample_path, 'wb') as f:
        pickle.dump(source_sample_list, f)
    return source_sample_path, target_sample_path


def random_sample_target(target_list, target_budget, save_path):
    random.shuffle(target_list)
    target_sample_list = random.sample(target_list, target_budget)
    target_sample_path = save_path / ('random_target_list.pkl')
    with open(target_sample_path, 'wb') as f:
        pickle.dump(target_sample_list, f)
    return target_sample_path
