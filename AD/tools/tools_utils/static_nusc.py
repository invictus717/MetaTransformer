import pickle
import numpy as np
import pandas as pd


with open('nuscenes_infos_10sweeps_train.pkl', 'rb') as f:
    nusc_info = pickle.load(f)

nusc_car = None
nusc_ped = None
nusc_cyc = None
for i, item in enumerate(nusc_info):
    gt_boxes = item['gt_boxes']
    gt_names = item['gt_names']
    mask_car = gt_names == 'car'
    mask_ped = gt_names == 'pedestrian'
    mask_cyc = gt_names == 'bicycle'
    car_info = gt_boxes[mask_car]
    ped_info = gt_boxes[mask_ped]
    cyc_info = gt_boxes[mask_cyc]

    if i == 0:
        nusc_car = car_info
        nusc_ped = ped_info
        nusc_cyc = cyc_info
    else:
        try:
            nusc_car = np.concatenate([nusc_car, car_info], axis=0)
        except:
            pass
        try:
            nusc_ped = np.concatenate([nusc_ped, ped_info], axis=0)
        except:
            pass
        try:
            nusc_cyc = np.concatenate([nusc_cyc, cyc_info], axis=0)
        except:
            pass

print('car_num: %d' % len(nusc_car))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_car[:, 2]), np.std(nusc_car[:, 2]), np.min(nusc_car[:, 2]), np.max(nusc_car[:, 2]), np.median(nusc_car[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_car[:, 3]), np.std(nusc_car[:, 3]), np.min(nusc_car[:, 3]), np.max(nusc_car[:, 3]), np.median(nusc_car[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_car[:, 4]), np.std(nusc_car[:, 4]), np.min(nusc_car[:, 4]), np.max(nusc_car[:, 4]), np.median(nusc_car[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_car[:, 5]), np.std(nusc_car[:, 5]), np.min(nusc_car[:, 5]), np.max(nusc_car[:, 5]), np.median(nusc_car[:, 5])))

print('ped_num: %d' % len(nusc_ped))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_ped[:, 2]), np.std(nusc_ped[:, 2]), np.min(nusc_ped[:, 2]), np.max(nusc_ped[:, 2]), np.median(nusc_ped[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_ped[:, 3]), np.std(nusc_ped[:, 3]), np.min(nusc_ped[:, 3]), np.max(nusc_ped[:, 3]), np.median(nusc_ped[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_ped[:, 4]), np.std(nusc_ped[:, 4]), np.min(nusc_ped[:, 4]), np.max(nusc_ped[:, 4]), np.median(nusc_ped[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_ped[:, 5]), np.std(nusc_ped[:, 5]), np.min(nusc_ped[:, 5]), np.max(nusc_ped[:, 5]), np.median(nusc_ped[:, 5])))

print('bicycle_num: %d' % len(nusc_cyc))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_cyc[:, 2]), np.std(nusc_cyc[:, 2]), np.min(nusc_cyc[:, 2]), np.max(nusc_cyc[:, 2]), np.median(nusc_cyc[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_cyc[:, 3]), np.std(nusc_cyc[:, 3]), np.min(nusc_cyc[:, 3]), np.max(nusc_cyc[:, 3]), np.median(nusc_cyc[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_cyc[:, 4]), np.std(nusc_cyc[:, 4]), np.min(nusc_cyc[:, 4]), np.max(nusc_cyc[:, 4]), np.median(nusc_cyc[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(nusc_cyc[:, 5]), np.std(nusc_cyc[:, 5]), np.min(nusc_cyc[:, 5]), np.max(nusc_cyc[:, 5]), np.median(nusc_cyc[:, 5])))

nusc_car_df = pd.DataFrame(nusc_car[:, 0:7], columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
nusc_ped_df = pd.DataFrame(nusc_ped[:, 0:7], columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
nusc_cyc_df = pd.DataFrame(nusc_cyc[:, 0:7], columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
nusc_car_df.to_csv('nuscenes_car.csv')
nusc_ped_df.to_csv('nuscenes_ped.csv')
nusc_cyc_df.to_csv('nuscenes_bicycle.csv')


# print(nusc_info[26])