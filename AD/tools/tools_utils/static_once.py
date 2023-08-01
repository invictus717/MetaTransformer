import pickle
import numpy as np
import pandas as pd
import json

once_car = None
once_truck = None
once_bus = None
once_veh = None
once_cyc = None
once_ped = None

with open('./once_infos_train.pkl', 'rb') as f:
    once_train_info = pickle.load(f)
json_str = json.dumps(once_train_info[5])
with open('./example.json', 'w') as f:
    f.write(json_str)

with open('./once_infos_val.pkl', 'rb') as f:
    once_val_info = pickle.load(f)

once_train_info = once_train_info + once_val_info

Veh = ['Car', 'Truck', 'Bus']
num = 0
for i, item in enumerate(once_train_info):
    try:
        gt_boxes = item['annos']['boxes_3d']
        gt_names = item['annos']['name']
    except:
        continue
    num = num + 1
    mask_car = gt_names == 'Car'
    mask_truck = gt_names == 'Truck'
    mask_bus = gt_names == 'Bus'
    mask_cyc = gt_names == 'Cyclist'
    mask_ped = gt_names == 'Pedestrian'
    
    mask_veh = []
    for j in range(len(gt_names)):
        if gt_names[j] in Veh:
            mask_veh.append(True)
        else:
            mask_veh.append(False)


    car_info = gt_boxes[mask_car]
    truck_info = gt_boxes[mask_truck]
    bus_info = gt_boxes[mask_bus]
    cyc_info = gt_boxes[mask_cyc]
    ped_info = gt_boxes[mask_ped]

    veh_info = gt_boxes[mask_veh]

    if i == 0:
        once_car = car_info
        once_truck = truck_info
        once_bus = bus_info
        once_cyc = cyc_info
        once_ped = ped_info
        once_veh = veh_info
    
    else:
        try:
            once_car = np.concatenate([once_car, car_info], axis=0)
        except:
            pass
        try:
            once_truck = np.concatenate([once_truck, truck_info], axis=0)
        except:
            pass
        try:
            once_bus = np.concatenate([once_bus, bus_info], axis=0)
        except:
            pass
        try:
            once_cyc = np.concatenate([once_cyc, cyc_info], axis=0)
        except:
            pass
        try:
            once_ped = np.concatenate([once_ped, ped_info], axis=0)
        except:
            pass
        try:
            once_veh = np.concatenate([once_veh, veh_info], axis=0)
        except:
            pass

print(num)

print('car_num: %d' % len(once_car))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_car[:, 2]), np.std(once_car[:, 2]), np.min(once_car[:, 2]), np.max(once_car[:, 2]), np.median(once_car[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_car[:, 3]), np.std(once_car[:, 3]), np.min(once_car[:, 3]), np.max(once_car[:, 3]), np.median(once_car[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_car[:, 4]), np.std(once_car[:, 4]), np.min(once_car[:, 4]), np.max(once_car[:, 4]), np.median(once_car[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_car[:, 5]), np.std(once_car[:, 5]), np.min(once_car[:, 5]), np.max(once_car[:, 5]), np.median(once_car[:, 5])))

print('truck_num: %d' % len(once_truck))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_truck[:, 2]), np.std(once_truck[:, 2]), np.min(once_truck[:, 2]), np.max(once_truck[:, 2]), np.median(once_truck[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_truck[:, 3]), np.std(once_truck[:, 3]), np.min(once_truck[:, 3]), np.max(once_truck[:, 3]), np.median(once_truck[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_truck[:, 4]), np.std(once_truck[:, 4]), np.min(once_truck[:, 4]), np.max(once_truck[:, 4]), np.median(once_truck[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_truck[:, 5]), np.std(once_truck[:, 5]), np.min(once_truck[:, 5]), np.max(once_truck[:, 5]), np.median(once_truck[:, 5])))

print('bus_num: %d' % len(once_bus))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_bus[:, 2]), np.std(once_bus[:, 2]), np.min(once_bus[:, 2]), np.max(once_bus[:, 2]), np.median(once_bus[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_bus[:, 3]), np.std(once_bus[:, 3]), np.min(once_bus[:, 3]), np.max(once_bus[:, 3]), np.median(once_bus[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_bus[:, 4]), np.std(once_bus[:, 4]), np.min(once_bus[:, 4]), np.max(once_bus[:, 4]), np.median(once_bus[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_bus[:, 5]), np.std(once_bus[:, 5]), np.min(once_bus[:, 5]), np.max(once_bus[:, 5]), np.median(once_bus[:, 5])))

print('ped_num: %d' % len(once_ped))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_ped[:, 2]), np.std(once_ped[:, 2]), np.min(once_ped[:, 2]), np.max(once_ped[:, 2]), np.median(once_ped[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_ped[:, 3]), np.std(once_ped[:, 3]), np.min(once_ped[:, 3]), np.max(once_ped[:, 3]), np.median(once_ped[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_ped[:, 4]), np.std(once_ped[:, 4]), np.min(once_ped[:, 4]), np.max(once_ped[:, 4]), np.median(once_ped[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_ped[:, 5]), np.std(once_ped[:, 5]), np.min(once_ped[:, 5]), np.max(once_ped[:, 5]), np.median(once_ped[:, 5])))

print('cyc_num: %d' % len(once_cyc))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_cyc[:, 2]), np.std(once_cyc[:, 2]), np.min(once_cyc[:, 2]), np.max(once_cyc[:, 2]), np.median(once_cyc[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_cyc[:, 3]), np.std(once_cyc[:, 3]), np.min(once_cyc[:, 3]), np.max(once_cyc[:, 3]), np.median(once_cyc[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_cyc[:, 4]), np.std(once_cyc[:, 4]), np.min(once_cyc[:, 4]), np.max(once_cyc[:, 4]), np.median(once_cyc[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_cyc[:, 5]), np.std(once_cyc[:, 5]), np.min(once_cyc[:, 5]), np.max(once_cyc[:, 5]), np.median(once_cyc[:, 5])))

print('veh_num: %d' % len(once_veh))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_veh[:, 2]), np.std(once_veh[:, 2]), np.min(once_veh[:, 2]), np.max(once_veh[:, 2]), np.median(once_veh[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_veh[:, 3]), np.std(once_veh[:, 3]), np.min(once_veh[:, 3]), np.max(once_veh[:, 3]), np.median(once_veh[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_veh[:, 4]), np.std(once_veh[:, 4]), np.min(once_veh[:, 4]), np.max(once_veh[:, 4]), np.median(once_veh[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(once_veh[:, 5]), np.std(once_veh[:, 5]), np.min(once_veh[:, 5]), np.max(once_veh[:, 5]), np.median(once_veh[:, 5])))


once_car_df = pd.DataFrame(once_car, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
once_car_df.to_csv('once_car.csv')
once_truck_df = pd.DataFrame(once_truck, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
once_truck_df.to_csv('once_truck.csv')
once_bus_df = pd.DataFrame(once_bus, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
once_bus_df.to_csv('once_bus.csv')
once_ped_df = pd.DataFrame(once_ped, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
once_ped_df.to_csv('once_pedestrian.csv')
once_cyc_df = pd.DataFrame(once_cyc, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
once_cyc_df.to_csv('once_cyclist.csv')
