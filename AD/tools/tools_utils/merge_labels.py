import pickle
from re import L
from turtle import st
import numpy as np
import argparse

def main(args):

    assert args.raw_data_pkl != None, 'raw_data path cannot be None'
    with open(args.raw_data_pkl, 'rb') as f:
        raw_data_info = pickle.load(f)

    class_names = []
    
    if args.vehicle_pkl:
        with open(args.vehicle_pkl, 'rb') as f:
            vehicle_result = pickle.load(f)
            class_names.append('Vehicle')
            assert len(vehicle_result) == len(raw_data_info), 'Vehicle file and raw data file are not corresponded'        
    else:
        print('++ No vehicle pseudo info.')     

    if args.cyclist_pkl:
        with open(args.cyclist_pkl, 'rb') as f:
            cyclist_result = pickle.load(f)
            class_names.append('Cyclist')
            assert len(cyclist_result) == len(raw_data_info), 'Cyclist file and raw data file are not corresponded'
    else:
        print('++ No cyclist pseudo info.')

    if args.pedestrian_pkl:
        with open(args.pedestrian_pkl, 'rb') as f:
            pedestrian_result = pickle.load(f)
            class_names.append('Pedestrian')
            assert len(pedestrian_result) == len(raw_data_info), 'Pedestrian file and raw data file are not corresponded'
    else:
        print('++ No pedestrian pseudo info.')

    vehi_num = 0 
    cyc_num = 0 
    pede_num = 0

    for i, raw_data in enumerate(raw_data_info):
        if 'Vehicle' in class_names:
            veh = vehicle_result[i]
            assert veh['frame_id'] == raw_data['frame_id']
            gt_mask = veh['name'] == 'Vehicle'
            vehi_num = vehi_num + np.sum(gt_mask!=0)
            gt_names_veh = list(veh['name'][gt_mask])
            gt_boxes_veh = list(veh['boxes_3d'][gt_mask])
            gt_score_veh = list(veh['score'][gt_mask])
        else:
            gt_names_veh = []
            gt_boxes_veh = []
            gt_score_veh = []

        if 'Cyclist' in class_names:
            cyc = cyclist_result[i]
            assert cyc['frame_id'] == raw_data['frame_id']
            gt_mask = cyc['name'] == 'Cyclist'
            cyc_num = cyc_num + np.sum(gt_mask!=0)
            gt_names_cyc = list(cyc['name'][gt_mask])
            gt_boxes_cyc = list(cyc['boxes_3d'][gt_mask])
            gt_score_cyc = list(cyc['score'][gt_mask])
        else:
            gt_names_cyc = []
            gt_boxes_cyc = []
            gt_score_cyc = []

        if 'Pedestrian' in class_names:
            ped = pedestrian_result[i]
            assert ped['frame_id'] == raw_data['frame_id']
            gt_mask = ped['name'] == 'Pedestrian'
            pede_num = pede_num + np.sum(gt_mask!=0)
            gt_names_ped = list(ped['name'][gt_mask])
            gt_boxes_ped = list(ped['boxes_3d'][gt_mask])
            gt_score_ped = list(ped['score'][gt_mask])
        else:
            gt_names_ped = []
            gt_boxes_ped = []
            gt_score_ped = []


        gt_names = np.array(gt_names_veh + gt_names_cyc + gt_names_ped)
        gt_boxes = np.array(gt_boxes_veh + gt_boxes_cyc + gt_boxes_ped, dtype=np.float64)
        gt_scores = np.array(gt_score_veh + gt_score_cyc + gt_score_ped)

        if gt_names.size == 0:
            continue
        else:
            annos = {
                'name': gt_names,
                'boxes_3d': gt_boxes,
                'boxes_score': gt_scores
            }
            raw_data.update({'annos': annos})
    with open(args.save_path, 'wb') as f:
        pickle.dump(raw_data_info, f)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--vehicle_pkl', type=str, default='')
    parser.add_argument('--cyclist_pkl', type=str, default='')
    parser.add_argument('--pedestrian_pkl',type=str, default='')
    parser.add_argument('--raw_data_pkl', type=str, default='')
    parser.add_argument('--save_path', type=str, default='test_1.pkl')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    main(args)