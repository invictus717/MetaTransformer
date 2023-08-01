import os
import torch
import pickle
import json

location_info_path = ""

nuscenes_info_path_train = ""
nuscenes_info_path_val   = ""

with open(nuscenes_info_path_train, 'rb') as f:
    infos_train = pickle.load(f)

with open(nuscenes_info_path_val, 'rb') as f:
    infos_val = pickle.load(f)



with open(location_info_path, 'rb') as f:
    location_info = json.load(f)

token2location = {}

for info in location_info:
    token2location[info['logfile']] = info['location']

location2token = {}
for token in token2location.keys():
    if token2location[token] not in location2token.keys():
        location2token[token2location[token]] = []
    location2token[token2location[token]].append(token)

singapore_onenorth_list_train = []
boston_seaport_list_train = []
singapore_queenstown_list_train = []
singapore_hollandvillage_list_train = []


for info in infos_train:
    token = info['cam_front_path'].split('/')[-1].split('_')[0]
    location = token2location[token]
    if location == 'singapore-onenorth':
        singapore_onenorth_list_train.append(info)
    elif location == 'boston-seaport':
        boston_seaport_list_train.append(info)
    elif location =='singapore-queenstown':
        singapore_queenstown_list_train.append(info)
    elif location == 'singapore-hollandvillage':
        singapore_hollandvillage_list_train.append(info)



with open('singapore-onenorth_data_train.pkl', 'wb') as f:
    pickle.dump(singapore_onenorth_list_train, f)

with open('boston-seaport_data_train.pkl', 'wb') as f:
    pickle.dump(boston_seaport_list_train, f)

with open('singapore-queenstown_data_train.pkl', 'wb') as f:
    pickle.dump(singapore_queenstown_list_train, f)

with open('singapore-hollandvillage_data_train.pkl', 'wb') as f:
    pickle.dump(singapore_hollandvillage_list_train, f)

singapore_onenorth_list_val = []
boston_seaport_list_val = []
singapore_queenstown_list_val = []
singapore_hollandvillage_list_val = []

for info in infos_val:
    token = info['cam_front_path'].split('/')[-1].split('_')[0]
    location = token2location[token]
    if location == 'singapore-onenorth':
        singapore_onenorth_list_val.append(info)
    elif location == 'boston-seaport':
        boston_seaport_list_val.append(info)
    elif location =='singapore-queenstown':
        singapore_queenstown_list_val.append(info)
    elif location == 'singapore-hollandvillage':
        singapore_hollandvillage_list_val.append(info)

with open('singapore-onenorth_data_val.pkl', 'wb') as f:
    pickle.dump(singapore_onenorth_list_val, f)

with open('boston-seaport_data_val.pkl', 'wb') as f:
    pickle.dump(boston_seaport_list_val, f)

with open('singapore-queenstown_data_val.pkl', 'wb') as f:
    pickle.dump(singapore_queenstown_list_val, f)

with open('singapore-hollandvillage_data_val.pkl', 'wb') as f:
    pickle.dump(singapore_hollandvillage_list_val, f)

    
print('singapore_onenorth_list_train:', len(singapore_onenorth_list_train))
print('singapore_onenorth_list_val:', len(singapore_onenorth_list_val))
print('boston_seaport_list_train:', len(boston_seaport_list_train))
print('boston_seaport_list_val', len(boston_seaport_list_val))
print('singapore_queenstown_list_train:', len(singapore_queenstown_list_train))
print('singapore_queenstown_list_val:', len(singapore_queenstown_list_val))
print('singapore_hollandvillage_list_train:', len(singapore_hollandvillage_list_train))
print('singapore_hollandvillage_list_val:', len(singapore_hollandvillage_list_val))

# print(len(infos_train) + len(infos_val))
# print(len(singapore_onenorth_list)+len(boston_seaport_list)+len(singapore_queenstown_list)+len(singapore_hollandvillage_list))