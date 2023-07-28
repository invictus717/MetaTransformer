import torch
import numpy as np 
from utils import *
import argparse
from data.dataset import data_loader, data_loader_attacks

parser = argparse.ArgumentParser(description='ROC For Attack')

parser.add_argument('--clean_image_folder_path', type=str , 
                    help='Path to root directory of images')

parser.add_argument('--attack_image_folder_path', type=str , 
                    help='Path to root directory of images')

parser.add_argument('--vit_path', type=str , 
                    help='Path to the downloaded ViT model')

parser.add_argument('--mlp_path', type=str , 
                    help='Path to the downloaded MLPs folder')

parser.add_argument('--attack_name', type=str,
                    help='Attack name')

args = parser.parse_args()

#Load Models 
model = torch.load(args.vit_path).cuda()
model.eval()
print('ViT is loaded!')

#Load MLPs
MLPs_list = get_classifiers_list(MLP_path=args.mlp_path)
print('All MLPs are loaded!')


#Load Images (clean and attack)
batch_size = 10
clean_loader_, _= data_loader(root_dir=args.clean_image_folder_path, batch_size=batch_size)
attack_loader_, _= data_loader_attacks(root_dir=args.attack_image_folder_path, attack_name= args.attack_name, batch_size=batch_size)
print('Clean test samples and corresponding adversarial samples are loaded')

#Find Frobenuis Norm
frob_list_clean = frobenius_norm(data_loader=clean_loader_['test'], model=model, mlps_list= MLPs_list)
frob_list_attack = frobenius_norm(data_loader=attack_loader_, model=model, mlps_list= MLPs_list)
print('Frobenuis norm has been calculated')


frob_dict = {'clean': torch.tensor(frob_list_clean), args.attack_name:torch.tensor(frob_list_attack)}
#Find TPR and FPR
tpr_list, fpr_list, threshold = roc(attack_name= args.attack_name, frob_dict= frob_dict, threshold= np.arange(0,90,0.1))
#Plot ROC
plot_roc(tpr_list= tpr_list, fpr_list= fpr_list, attack_name= args.attack_name)
print('ROC figure has been saved in the current directory!')