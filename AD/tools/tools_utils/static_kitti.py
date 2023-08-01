import pickle
import numpy as np
import pandas as pd


with open('kitti_infos_trainval.pkl', 'rb') as f:
    kitti_infos = pickle.load(f)

kitti_car = None
kitti_ped = None
kitti_cyc = None
for i, item in enumerate(kitti_infos):
    gt_info = item['annos']
    mask_dontcare = gt_info['name'] != 'DontCare'
    mask_car = gt_info['name'][mask_dontcare] == 'Car'
    mask_ped = gt_info['name'][mask_dontcare] == 'Pedestrian'
    mask_cyc = gt_info['name'][mask_dontcare] == 'Cyclist'
    car_info = gt_info['gt_boxes_lidar'][mask_car]
    ped_info = gt_info['gt_boxes_lidar'][mask_ped]
    cyc_info = gt_info['gt_boxes_lidar'][mask_cyc]

    if i == 0:
        kitti_car = car_info
        kitti_ped = ped_info
        kitti_cyc = cyc_info
    else:
        kitti_car = np.concatenate([kitti_car, car_info], axis=0)
        kitti_ped = np.concatenate([kitti_ped, ped_info], axis=0)
        kitti_cyc = np.concatenate([kitti_cyc, cyc_info], axis=0)


print('car_num: %d' % len(kitti_car))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_car[:, 2]), np.std(kitti_car[:, 2]), np.min(kitti_car[:, 2]), np.max(kitti_car[:, 2]), np.median(kitti_car[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_car[:, 3]), np.std(kitti_car[:, 3]), np.min(kitti_car[:, 3]), np.max(kitti_car[:, 3]), np.median(kitti_car[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_car[:, 4]), np.std(kitti_car[:, 4]), np.min(kitti_car[:, 4]), np.max(kitti_car[:, 4]), np.median(kitti_car[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_car[:, 5]), np.std(kitti_car[:, 5]), np.min(kitti_car[:, 5]), np.max(kitti_car[:, 5]), np.median(kitti_car[:, 5])))

print('ped_num: %d' % len(kitti_ped))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_ped[:, 2]), np.std(kitti_ped[:, 2]), np.min(kitti_ped[:, 2]), np.max(kitti_ped[:, 2]), np.median(kitti_ped[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_ped[:, 3]), np.std(kitti_ped[:, 3]), np.min(kitti_ped[:, 3]), np.max(kitti_ped[:, 3]), np.median(kitti_ped[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_ped[:, 4]), np.std(kitti_ped[:, 4]), np.min(kitti_ped[:, 4]), np.max(kitti_ped[:, 4]), np.median(kitti_ped[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_ped[:, 5]), np.std(kitti_ped[:, 5]), np.min(kitti_ped[:, 5]), np.max(kitti_ped[:, 5]), np.median(kitti_ped[:, 5])))

print('bicycle_num: %d' % len(kitti_cyc))
print('Z--------mean:%f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_cyc[:, 2]), np.std(kitti_cyc[:, 2]), np.min(kitti_cyc[:, 2]), np.max(kitti_cyc[:, 2]), np.median(kitti_cyc[:, 2])))
print('Length---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_cyc[:, 3]), np.std(kitti_cyc[:, 3]), np.min(kitti_cyc[:, 3]), np.max(kitti_cyc[:, 3]), np.median(kitti_cyc[:, 3])))
print('Width---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_cyc[:, 4]), np.std(kitti_cyc[:, 4]), np.min(kitti_cyc[:, 4]), np.max(kitti_cyc[:, 4]), np.median(kitti_cyc[:, 4])))
print('Height---mean: %f, std: %f, min: %f, max: %f, median: %f' % (np.mean(kitti_cyc[:, 5]), np.std(kitti_cyc[:, 5]), np.min(kitti_cyc[:, 5]), np.max(kitti_cyc[:, 5]), np.median(kitti_cyc[:, 5])))

kitti_car_df = pd.DataFrame(kitti_car, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
kitti_car_df.to_csv('kitti_ped.csv')
kitti_ped_df = pd.DataFrame(kitti_ped, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
kitti_ped_df.to_csv('kitti_ped.csv')
kitti_cyc_df = pd.DataFrame(kitti_cyc, columns=['center_x', 'center_y', 'center_z', 'L', 'W', 'H', 'angle'])
kitti_cyc_df.to_csv('kitti_cyc.csv')
