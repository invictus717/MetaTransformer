import os
import pickle
import io
from pathlib import Path
from petrel_client.client import Client
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io as sk_io


client = Client("~/.petreloss.conf")

def list_oss_dir(oss_path, with_info=False):
    files_iter = client.get_file_iterator(oss_path)
    if with_info:
        file_list = {p: k for p, k in files_iter}
    else:
        file_list = [p for p, k in files_iter]
    return file_list

def load_pkl_oss(oss_path):
    pkl_bytes = client.get(oss_path)
    infos = pickle.load(io.BytesIO(pkl_bytes))
    return infos

def splite_bbox(list_bbox):
    # should mainly calculate the z statistics
    # for kitti: x-y-z-l-w-h
    # for waymo: x-y-z-l-w-h
    bbox_z = []
    bbox_l = []
    bbox_w = []
    bbox_h = []
    for bbox in list_bbox:
        bbox_z.append(bbox[2])
        bbox_l.append(bbox[3])
        bbox_w.append(bbox[4])
        bbox_h.append(bbox[5])
    
    bbox_z_np = np.array(bbox_z)
    bbox_l_np = np.array(bbox_l)
    bbox_w_np = np.array(bbox_w)
    bbox_h_np = np.array(bbox_h)

    return bbox_z_np, bbox_l_np, bbox_h_np, bbox_w_np
    
def process_object_info(class_info, cls_name=None, get_abnorm_idx=False):
    info_order = ["z", "l", "h", "w"]
    statis = {}
    for idx, element in enumerate(splite_bbox(class_info)):
        if cls_name is not None:
            print(f"Process the class: {cls_name}")
        print(f"Current Process the Information along: {info_order[idx]}")
        statis[info_order[idx]] = get_statistic(element, get_abnorm=get_abnorm_idx)
        # draw_hist(element)
    return statis

def get_statistic(arr, get_abnorm=False):
    mean_arr = np.round(np.mean(arr), decimals=2)
    median_arr = np.round(np.median(arr), decimals=2)
    std_arr = np.round(np.std(arr), decimals=2)
    min_arr = np.round(np.min(arr), decimals=2)
    max_arr = np.round(np.max(arr), decimals=2)
    print(f"mean: {mean_arr}, std: {std_arr}, min: {min_arr}, max {max_arr}, median {median_arr}")
    statis = {}
    statis = {"mean":mean_arr, "std": std_arr, "min": min_arr, "max": max_arr, "median": median_arr}
    if get_abnorm:
        abnorm_min = mean_arr - 3 * std_arr
        abnorm_max = mean_arr + 3 * std_arr
        abnorm_max_index = np.where(arr > abnorm_max)
        abnorm_min_index = np.where(arr < abnorm_min)
        abnorm_idx_list = list(abnorm_max_index) + list(abnorm_min_index)
        statis["abnorm_obj_idx"] = abnorm_idx_list
        return statis
    return statis

def get_image(root_data_path, idx):
    """
    Loads image for a sample
    Args:
        idx: int, Sample index
    Returns:
        image: (H, W, 3), RGB Image
    """
    img_file = root_data_path + 'image_2'+ str('%s.png' % idx)
    # assert img_file.exists(), f"Image path {img_file} not exists"
    print(f"Try to load image: {img_file}")
    image = sk_io.imread(img_file)
    image = image.astype(np.float32)
    image /= 255.0
    return image


def draw_hist(a, num_bins=20):
    plt.figure(figsize=(20,8),dpi=80)
    plt.hist(a,num_bins,density=True)
    plt.grid(alpha=0.1)
    plt.show()


def add_rect_to_image(car_image_loc, abnorm_flag=False):
    # print(f"Car loc in func is: {car_image_loc}" )
    width = car_image_loc[2] - car_image_loc[0]
    height = car_image_loc[3] - car_image_loc[1]
    center = (car_image_loc[0], car_image_loc[1])
    if abnorm_flag:
        rect = patches.Rectangle(center, width, height, linewidth=2, edgecolor='r', facecolor='none')
    else:
        rect = patches.Rectangle(center, width, height, linewidth=1, edgecolor='g', facecolor='none')
    # print(f"the start: {center}, width: {width} and height: {height}")
    return rect


def kitti_process(abnorm_info_types=["z"]):
    kitti_path = #PATH TO DATASET
    bbox_info_pointer = {"x":0, "y":1, "z":2, "l":3, "w":4, "h":5}

    kitti_infos = load_pkl_oss(kitti_path)

    kitti_classes = ['Car','Pedestrian', 'Cyclist']
    kitti_car_info = []
    kitti_car_info_image = []
    kitti_car_frameIdx = []
    kitti_car_info_index = []
    kitti_ped_info = []
    kitti_cyc_info = []
    kitti_idx_list = []
    kitti_frame_car_counter = []
    kitti_info_class = {}
    frame_cnt = len(kitti_infos)
    for idx, info in enumerate(kitti_infos):
        lidar_idx = info["point_cloud"]["lidar_idx"]
        kitti_idx_list.append(lidar_idx)
        anno_info = info["annos"]
        obj_number = anno_info["name"].shape[0]
        car_counter = 0
        for i in range(obj_number):
            if anno_info["name"][i] == "Pedestrian":
                kitti_ped_info.append(anno_info["gt_boxes_lidar"][i])
            elif anno_info["name"][i] == "Car":
                car_counter += 1
                kitti_car_info.append(anno_info["gt_boxes_lidar"][i])
                kitti_car_info_image.append(anno_info['bbox'][i])
            elif anno_info["name"][i] == "Cyclist":
                kitti_cyc_info.append(anno_info["gt_boxes_lidar"][i])
            else:
                continue
        # used to fetch image/lidar files
        kitti_car_frameIdx.extend([lidar_idx] * car_counter)
        # uesd to fetch frame info
        kitti_car_info_index.extend([idx] * car_counter)
        kitti_frame_car_counter.extend([car_counter] *car_counter)

    kitti_info_class = {"car": kitti_car_info, "ped": kitti_ped_info, "cyc": kitti_cyc_info, "car_frameIdx":kitti_car_frameIdx}
    print(f"The totoal frame cout: {frame_cnt}")
    print(f"Car Counts: {len(kitti_car_info)}, Ped: {len(kitti_ped_info)}, Cyc: {len(kitti_cyc_info)}")
    assert len(kitti_car_info_index) == len(kitti_car_frameIdx) == len(kitti_frame_car_counter)

    for cls in kitti_info_class.keys():
        cls_info = kitti_info_class[cls]
        if cls != "car":
            continue
        print(f"Current Process {cls}")
        statis = process_object_info(cls_info, cls_name=cls, get_abnorm_idx=True)

    
    max_min_order = {"max": 0, "min": 1}
    for abnorm_info_type in abnorm_info_types:
        statis_info = statis[abnorm_info_type]

        kitti_image_save_path = ""
        
        for limit in max_min_order.keys():
            cur_save_path = os.path.join(kitti_image_save_path, abnorm_info_type, limit)
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
                print(f"Make folder: {cur_save_path}")
            car_info_image_abnorm = [kitti_car_info_image[k] for k in statis_info['abnorm_obj_idx'][max_min_order[limit]]]
            for idx, abnorm_idx in enumerate(statis_info['abnorm_obj_idx'][max_min_order[limit]]):
                # 0: 偏大 1： 偏小
                # abnorm_idx = 10
                abnorm_frame_idx = kitti_car_frameIdx[abnorm_idx]
                abnorm_info_idx = kitti_car_info_index[abnorm_idx]
                car_image_loc = car_info_image_abnorm[idx]
                ori_image_path = #PATH_TO_DATASET
                print(f"Load image from {ori_image_path}")
                
                image_name = kitti_infos[abnorm_info_idx]["point_cloud"]["lidar_idx"] + ".png"
                full_path = os.path.join(cur_save_path, image_name)
                # if os.path.exists(full_path):
                #     continue

                image_bytes = client.get(ori_image_path)
                image_npy = sk_io.imread(io.BytesIO(image_bytes))

                plt.figure(figsize=(30, 15))
                fig, ax = plt.subplots()

                # print(f"Car loc is: {car_image_loc}" )
                rect = add_rect_to_image(car_image_loc, abnorm_flag=True)
                ax.add_patch(rect)

                all_car_images = kitti_infos[abnorm_info_idx]["annos"]["bbox"]
                all_car_lidar = kitti_infos[abnorm_info_idx]["annos"]["gt_boxes_lidar"]
                car_counter = 0
                for type_ in kitti_infos[abnorm_info_idx]["annos"]["name"]:
                    if type_ == 'Car':
                        car_counter += 1
                for i in range(car_counter):
                    car_loc = all_car_images[i]
                    # print(f"Car loc in all bbox is: {car_loc}" )
                    rect_ = add_rect_to_image(car_loc, abnorm_flag=False)
                    plt.text(car_loc[0], car_loc[1], str(round(all_car_lidar[i][bbox_info_pointer[abnorm_info_type]],2)))
                    ax.add_patch(rect_)
                    # break

                ax.imshow(image_npy)
                # ax.add_image(image_npy)
                # plt.show()
                image_name = kitti_infos[abnorm_info_idx]["point_cloud"]["lidar_idx"] + ".png"
                full_path = os.path.join(cur_save_path, image_name)
                print(f"save abnorm statistic image to {full_path}")
                # plt.close(fig)
                plt.savefig(full_path)


if __name__ == "__main__":
    kitti_process(abnorm_info_types=["z", "l", "h", "w"])
