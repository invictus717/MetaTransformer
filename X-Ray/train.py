import argparse
import torch
from tqdm import tqdm
import os
from models import mlp
from data.dataset import data_loader
import torch.nn as nn

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


def train_vit(model, optimizer, data_loader_train, data_loader_test, num_epochs):
    """
    This function used to test ViT. 

    Args: 
        model: ViT model
        dataaloader_test: loader for test images 
    
    return: 
        Avg test accuracy of ViT
    
    """
    test_acc = 0.0
    for epoch in range(num_epochs):
        for images, labels in tqdm(data_loader_train): 
            optimizer.zero_grad()
            images = images.cuda()
            labels= labels.cuda()
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
        
        test_vit(model, data_loader_test)
        model.train()

    return None

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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--gamma', type=float, default=0.9)

args = parser.parse_args()

loader_, dataset_ = data_loader(root_dir=root_dir)



import timm
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.cuda()
# create timm 

from timm.models.vision_transformer import Block
import torch.nn as nn
ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
encoder.load_state_dict(ckpt,strict=True)
model.blocks = encoder

for n, p in model.blocks.named_parameters():
    # if 'adapter' not in n:
    p.requires_grad = False
trainables = [p for p in model.parameters() if p.requires_grad]
print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//10, gamma=args.gamma)

train_vit(model, optimizer, loader_['train'], loader_['test'], args.epochs)

trainables = [p for p in model.parameters() if p.requires_grad]
print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

# if args.model_name == 'ViT':
#     acc = test_vit(model=model, dataloader_test=loader_['test'])
# else:
#     mlps_list = sorted(os.listdir(args.mlp_path))
#     acc = test_mlps(mlps_list= mlps_list, dataloader_test=loader_['test'], mlp_root_dir=args.mlp_path)