import argparse
import torch
from tqdm import tqdm
import os
from models import mlp
from data.dataset import data_loader


root_dir = "./data/TB_data"


def test_vit(model, dataloader_test):
    """
    This function used to test ViT. 

    Args: 
        model: ViT model
        dataaloader_test: loader for test images 
    
    return: 
        Avg test accuracy of ViT
    
    """
    test_acc = 0.0
    for images, labels in tqdm(dataloader_test): 
        images = images.cuda()
        labels= labels.cuda()
        with torch.no_grad(): 
            model.eval()
            output = model(images)
            prediction = torch.argmax(output, dim=-1)
            acc = sum(prediction == labels).float().item()/len(labels)
            test_acc += acc 
    print(f'Testing accuracy = {(test_acc/len(dataloader_test)):.4f}')

    return round(test_acc/len(dataloader_test),2)

def test_mlps(mlps_list, dataloader_test, mlp_root_dir):
    for mlp in range(1, len(mlps_list) +1):
        acc_avg = 0.0
        mlp_in = torch.load(os.path.join(mlp_root_dir, mlps_list[mlp-1])).cuda()
        mlp_in.eval()
        print(f'MLP of index {mlp-1} has been loaded')

        for images, labels in tqdm(dataloader_test): 
            images = images.cuda()
            labels= labels.cuda()
            x = model.patch_embed(images)
            x = model.pos_drop(x)
            for block in range(mlp):
                x = model.blocks[block](x)
            with torch.no_grad():
                output = mlp_in(x)
            predictions = torch.argmax(output, dim=-1)
            acc = torch.sum(predictions == labels).item()/len(labels)
            acc_avg += acc 
        print(f'Accuracy of block {mlp-1} = {(acc_avg/len(dataloader_test)):.3f}')
            
    pass

parser = argparse.ArgumentParser(description='Testing ViT or MLPs')

parser.add_argument('--model_name', type=str , choices=['ViT','MLPs'],
                    help='Choose between ViT or MLPs')
parser.add_argument('--vit_path', type=str ,
                    help='pass the path of downloaded ViT')
parser.add_argument('--mlp_path', type=str ,
                    help='pass the path for the downloaded MLPs folder')
args = parser.parse_args()

loader_, dataset_ = data_loader(root_dir=root_dir)

for k, v in loader_.items():
    print(k, len(v))
exit()

model = torch.load(args.vit_path).cuda()
model.eval()

if args.model_name == 'ViT':
    acc = test_vit(model=model, dataloader_test=loader_['test'])
else:
    mlps_list = sorted(os.listdir(args.mlp_path))
    acc = test_mlps(mlps_list= mlps_list, dataloader_test=loader_['test'], mlp_root_dir=args.mlp_path)