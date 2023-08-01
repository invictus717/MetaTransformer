import os
import torch
import pickle
import json
import random
import copy

nuscenes_info_path_train = ""
once_info_path_train = ""
kitti_info = ""

with open(once_info_path_train, 'rb') as f:
    infos_train = pickle.load(f)

# random.shuffle(infos_train)
total_len = len(infos_train)

N = 10
infos_train_enlarge = copy.deepcopy(infos_train)
for i in range (1, N):
    infos_train_enlarge.extend(infos_train)

list_01 = infos_train[:int(total_len*0.01)]
list_05 = infos_train[:int(total_len*0.05)]
list_10 = infos_train[:int(total_len*0.10)]

with open('01_once_infos_train_vehicle.pkl', 'wb') as f:
    pickle.dump(list_01, f)

with open('05_once_infos_train_vehicle.pkl', 'wb') as f:
    pickle.dump(list_05, f)

with open('10_once_infos_train_vehicle.pkl', 'wb') as f:
    pickle.dump(list_10, f)
