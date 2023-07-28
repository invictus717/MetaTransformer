import os 
import torch 
import argparse
import numpy as np

from utils import *
from data.dataset import data_loader, data_loader_attacks

import mlp

def majority_voting(data_loader, model, mlps_list):
    """
    SEViT performance with majority voting. 

    Args: 
    data_loader: loader of test samples for clean images, or attackes generated from the test samples
    model: ViT model 
    mlps_list: list of intermediate MLPs

    Return: 
    Accuracy. 

    """
    acc_ = 0.0 
    for images, labels in data_loader:
        final_prediction = []
        images = images.cuda()
        vit_output = model(images)
        vit_predictions = torch.argmax(vit_output.detach().cpu(), dim=-1)
        final_prediction.append(vit_predictions.detach().cpu())
        x = model.patch_embed(images)
        x_0 = model.pos_drop(x)
        i=0
        for mlp in mlps_list:
            x_0 = model.blocks[i](x_0)
            mlp_output = mlp(x_0)
            mlp_predictions = torch.argmax(mlp_output.detach().cpu(), dim=-1)
            final_prediction.append(mlp_predictions.detach().cpu())
            i+=1
        stacked_tesnor = torch.stack(final_prediction,dim=1)
        preds_major = torch.argmax(torch.nn.functional.one_hot(stacked_tesnor).sum(dim=1), dim=-1)
        acc = (preds_major == labels).sum().item()/len(labels)
        acc_ += acc
    final_acc = acc_ / len(data_loader)
    print(f'Final Accuracy From Majority Voting = {(final_acc *100) :.3f}%' )
    return final_acc



parser = argparse.ArgumentParser(description='Majority Voting')

parser.add_argument('--images_type', type=str , choices=['clean', 'adversarial'],
                    help='Path to root directory of images')

parser.add_argument('--image_folder_path', type=str , 
                    help='Path to root directory of images')

parser.add_argument('--vit_path', type=str , 
                    help='Path to the downloaded ViT model')

parser.add_argument('--mlp_path', type=str , 
                    help='Path to the downloaded MLPs folder')

parser.add_argument('--attack_name', type=str,
                    help='Attack name')

args = parser.parse_args()


model = torch.load(args.vit_path).cuda()
model.eval()
print('ViT is loaded!')

MLPs_list = get_classifiers_list(MLP_path=args.mlp_path)
print('All MLPs are loaded!')

if args.images_type == 'clean': 
    loader_, dataset_ = data_loader(root_dir=args.image_folder_path, batch_size=15)
    majority_voting(data_loader=loader_['test'], model= model, mlps_list=MLPs_list)
else: 
    loader_, dataset_ = data_loader_attacks(root_dir=args.image_folder_path, attack_name= args.attack_name, batch_size=15)
    majority_voting(data_loader=loader_, model= model, mlps_list=MLPs_list)



