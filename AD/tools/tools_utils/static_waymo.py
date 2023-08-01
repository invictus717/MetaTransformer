import pickle
import numpy as np
import pandas as pd


# with open('nuscenes_infos_10sweeps_train.pkl', 'rb') as f:
#     nusc_info = pickle.load(f)

# nusc_car = None
# for i, item in enumerate(nusc_info):
#     gt_boxes = item['gt_boxes']
#     gt_names = item['gt_names']
#     mask = gt_names == 'car'
#     car_info = gt_boxes[mask]
#     # print(car_info, car_info.shape)
#     # if i == 10:
#     #     break
#     if i == 0:
#         nusc_car = car_info
#     else:
#         try:
#             nusc_car = np.concatenate([nusc_car, car_info], axis=0)
#         except:
#             pass

# nusc_df = pd.DataFrame(nusc_car[:, 0:7], columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
# nusc_df.to_csv('nuscenes_car.csv')


# print(nusc_info[2616])

# with open('waymo_processed_data_v0_5_0_infos_train.pkl', 'rb') as f:
#     waymo_info = pickle.load(f)


# waymo_car = None

# for i, item in enumerate(waymo_info[::]):
#     gt_boxes = item['annos']['gt_boxes_lidar']

#     gt_names = item['annos']['name']
#     mask = gt_names == 'Vehicle'
#     car_info = gt_boxes[mask]
#     if i == 0:
#         waymo_car = car_info
#     else:
#         try:
#             waymo_car = np.concatenate([waymo_car, car_info], axis=0)
#         except:
#             pass

# waymo_df = pd.DataFrame(waymo_car, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
# waymo_df.to_csv('waymo_car.csv')

with open('waymo_processed_data_v0_5_0_infos_train.pkl', 'rb') as f:
    waymo_info = pickle.load(f)

waymo_car = None
waymo_ped = None
waymo_cyc = None

for i, item in enumerate(waymo_info):
    gt_boxes = item['annos']['gt_boxes_lidar']

    gt_names = item['annos']['name']
    mask_car = gt_names == 'Vehicle'
    mask_ped = gt_names == 'Pedestrian'
    mask_cyc = gt_names == 'Cyclist'
    car_info = gt_boxes[mask_car]
    ped_info = gt_boxes[mask_ped]
    cyc_info = gt_boxes[mask_cyc]
    if i == 0:
        waymo_car = car_info
        waymo_ped = ped_info
        waymo_cyc = cyc_info
    else:
        try:
            waymo_car = np.concatenate([waymo_car, car_info], axis=0)
        except:
            pass
        try:
            waymo_ped = np.concatenate([waymo_ped, ped_info], axis=0)
        except:
            pass
        try:
            waymo_cyc = np.concatenate([waymo_cyc, cyc_info], axis=0)
        except:
            pass

print('car_num: %d' % len(waymo_car))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_car[:, 2]), np.std(waymo_car[:, 2]), np.min(waymo_car[:, 2]), np.max(waymo_car[:, 2]), np.median(waymo_car[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_car[:, 3]), np.std(waymo_car[:, 3]), np.min(waymo_car[:, 3]), np.max(waymo_car[:, 3]), np.median(waymo_car[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_car[:, 4]), np.std(waymo_car[:, 4]), np.min(waymo_car[:, 4]), np.max(waymo_car[:, 4]), np.median(waymo_car[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_car[:, 5]), np.std(waymo_car[:, 5]), np.min(waymo_car[:, 5]), np.max(waymo_car[:, 5]), np.median(waymo_car[:, 5])))

print('ped_num: %d' % len(waymo_ped))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_ped[:, 2]), np.std(waymo_ped[:, 2]), np.min(waymo_ped[:, 2]), np.max(waymo_ped[:, 2]), np.median(waymo_ped[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_ped[:, 3]), np.std(waymo_ped[:, 3]), np.min(waymo_ped[:, 3]), np.max(waymo_ped[:, 3]), np.median(waymo_ped[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_ped[:, 4]), np.std(waymo_ped[:, 4]), np.min(waymo_ped[:, 4]), np.max(waymo_ped[:, 4]), np.median(waymo_ped[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_ped[:, 5]), np.std(waymo_ped[:, 5]), np.min(waymo_ped[:, 5]), np.max(waymo_ped[:, 5]), np.median(waymo_ped[:, 5])))

print('cyc_num: %d' % len(waymo_cyc))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_cyc[:, 2]), np.std(waymo_cyc[:, 2]), np.min(waymo_cyc[:, 2]), np.max(waymo_cyc[:, 2]), np.median(waymo_cyc[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_cyc[:, 3]), np.std(waymo_cyc[:, 3]), np.min(waymo_cyc[:, 3]), np.max(waymo_cyc[:, 3]), np.median(waymo_cyc[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_cyc[:, 4]), np.std(waymo_cyc[:, 4]), np.min(waymo_cyc[:, 4]), np.max(waymo_cyc[:, 4]), np.median(waymo_cyc[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(waymo_cyc[:, 5]), np.std(waymo_cyc[:, 5]), np.min(waymo_cyc[:, 5]), np.max(waymo_cyc[:, 5]), np.median(waymo_cyc[:, 5])))

waymo_car_df = pd.DataFrame(waymo_car, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
waymo_car_df.to_csv('waymo_vehicle.csv')
waymo_ped_df = pd.DataFrame(waymo_ped, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
waymo_ped_df.to_csv('waymo_pedestrian.csv')
waymo_cyc_df = pd.DataFrame(waymo_cyc, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
waymo_cyc_df.to_csv('waymo_cyclist.csv')
