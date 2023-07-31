
import os, sys, zipfile, h5py, numpy as np, multiprocessing
from pathlib import Path
from tqdm import tqdm 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from openpoints.dataset.utils3d import *
from openpoints.dataset.vis3d import *


def unzip_folders(data_dir, unzipped_path):
    for dirpath, dirs, files in os.walk(data_dir):	 
        for file in files:
            if file.endswith('zip'):
                file_path = os.path.join(dirpath, file)
                zip_file = zipfile.ZipFile(file_path)
                for names in zip_file.namelist():
                    zip_file.extract(names, unzipped_path)
                zip_file.close() 


def save_room_objects(mesh_path, save_name):
    """save the objects in a rooom together 

    Args:
        mesh_path (_type_): _description_
        save_name (_type_): _description_
    """
    all_points = []
    all_ins_labels = []
    all_cls_labels = []
    
    points, faces, f_cls_id, f_ins_id = read_mesh_vf(mesh_path)
    # category id: https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv 
    # vis_points(points[:, :3], points[:, 3:6])
    f_cls_id_unique = np.unique(f_cls_id)   # the unique category ids (eg. pillow, wall)
    for cls in f_cls_id_unique:
        face_cls_flag = f_cls_id==cls   # wheter face is current cls
        face_cls = faces[face_cls_flag] # pick the face which is current cls
        face_cls_ins = f_ins_id[face_cls_flag]  # pick the instance label of the face
        for ins_id in np.unique(face_cls_ins):  # for each instance 
            point_idx = face_cls[face_cls_ins==ins_id].reshape(-1)
            points_cls_ins = points[point_idx]
            all_points.append(points_cls_ins)
            all_ins_labels.append(ins_id)
            all_cls_labels.append(cls)
    data = {'points': all_points, 'ins_labels': all_ins_labels, 'y': all_cls_labels}
    np.save(f'{save_name}.npy', data)


def save_per_object(mesh_path, save_name):
    """save each object in a h5 file.  

    Args:
        mesh_path (_type_): _description_
        save_name (_type_): _description_
    """
    
    points, faces, f_cls_id, f_ins_id = read_mesh_vf(mesh_path)
    # category id: https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv 
    # vis_points(points[:, :3], points[:, 3:6])
    f_cls_id_unique = np.unique(f_cls_id)   # the unique category ids (eg. pillow, wall)
    for cls in f_cls_id_unique:
        face_cls_flag = f_cls_id==cls   # wheter face is current cls
        face_cls = faces[face_cls_flag] # pick the face which is current cls
        face_cls_ins = f_ins_id[face_cls_flag]  # pick the instance label of the face

        for ins_id in np.unique(face_cls_ins):  # for each instance 
            point_idx = face_cls[face_cls_ins==ins_id].reshape(-1)
            points_cls_ins = points[point_idx]
            vis_points(points_cls_ins[:, :3], points_cls_ins[:, 3:6])
            data = {'points': points_cls_ins, 'ins_label': ins_id, 'label': cls}
            
            # TODO: debug comment for now. 
            np.save(f'{save_name}_C{cls}_I{ins_id}.npy', data)
    
    
if __name__ == "__main__":
    # unzip data at first. 
    # root_path = '/data/3D/Matterport3d/v1/scans'
    # unzipped_path = '/data/3D/Matterport3d/v1/scans_unzipped'
    # unzip_folders(root_path, unzipped_path)
    
    # loading all points to a single h5 file
    
    root_path = Path('/data/3D/Matterport3d/v1/')
    
    raw_path = root_path.joinpath('scans_unzipped')
    save_dir = root_path.joinpath('objects')
    save_dir.mkdir(exist_ok=True)
    
    all_points = []
    all_ins_labels = []
    all_cls_labels = []
    
    # p = multiprocessing.Pool()
    for path in Path(raw_path).rglob('*.ply'):
        file_path = path.parent.joinpath(path.name)
        file_name = '/'.join(path.parts[-3:])
        save_name = '_'.join([path.parts[-3], path.parts[-1].split('.')[0]])
        save_path = os.path.join(save_dir, save_name)
        print(f'===> processing {file_name} ...')
        save_per_object(file_path, save_path)    
    #     p.apply_async(save_per_object, [file_path, save_path]) 
    # p.close()
    # p.join() # Wait for all child processes to close.
