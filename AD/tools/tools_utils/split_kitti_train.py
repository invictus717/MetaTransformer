import os
import torch
import pickle
import json
import copy 
import random

nuscenes_info_path_train = ""

with open(nuscenes_info_path_train, 'rb') as f:
    infos_train = pickle.load(f)

random.shuffle(infos_train)
total_len = len(infos_train)

# list_01 = infos_train[:int(total_len*0.01)]
list_05 = infos_train[:int(total_len*0.05)]
# list_10 = infos_train[:int(total_len*0.10)]
# list_25 = infos_train[:int(total_len*0.25)]
# list_50 = infos_train[:int(total_len*0.5)]
# list_75 = infos_train[:int(total_len*0.75)]

#list_700 = 6*infos_train

# with open('01_kitti_infos_train.pkl', 'wb') as f:
#     pickle.dump(list_01, f)

with open('05_kitti_infos_train.pkl', 'wb') as f:
    pickle.dump(list_05, f)

# with open('10_kitti_infos_train.pkl', 'wb') as f:
#     pickle.dump(list_10, f)

# with open('25_kitti_infos_train.pkl', 'wb') as f:
#     pickle.dump(list_25, f)

# with open('50_kitti_infos_train.pkl', 'wb') as f:
#     pickle.dump(list_50, f)

# with open('75_kitti_infos_train.pkl', 'wb') as f:
#     pickle.dump(list_75, f)

# with open('700_kitti_infos_train.pkl', 'wb') as f:
#      pickle.dump(list_700, f)
