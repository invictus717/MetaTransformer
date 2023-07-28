import torch 
import argparse

from attack import Attack
from utils import *
from data.dataset import data_loader

parser = argparse.ArgumentParser(description='Generate Attack from ViT')

parser.add_argument('--epsilons', type=float , 
                    help='Perturbations Size')
parser.add_argument('--attack_list', type=str , nargs='+',
                    help='Attack List to Generate')
parser.add_argument('--vit_path', type=str ,
                    help='pass the path for the downloaded MLPs folder')
parser.add_argument('--attack_images_dir', type=str ,
                    help='Directory to save the generated attacks')
args = parser.parse_args()

root_dir = "./data/TB_data"
loader_, dataset_ = data_loader(root_dir=root_dir)

model = torch.load(args.vit_path).cuda()
model.eval()


#Generate and save attacks
generate_save_attacks(
    attack_names= args.attack_list,
    model= model,
    samples= loader_['test'], 
    classes= ['Normal', 'Tuberculosis'],
    attack_image_dir= args.attack_images_dir,
    epsilon=args.epsilons,
)
