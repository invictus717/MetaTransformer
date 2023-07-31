import torch
import torch.nn as nn
# from knn_cuda import KNN
from ..build import MODELS
from openpoints.utils.logger import *
from ..layers import trunc_normal_, DropPath, fps, SubsampleGroup, TransformerEncoder
from openpoints.utils.ckpt_util import get_missing_parameters_message, get_unexpected_parameters_message


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, 
                 in_chans=3, num_classes=40,
                 embed_dim=768, depth=12,
                 num_heads=12, 
                #  mlp_ratio=4., qkv_bias=False,
                #  drop_rate=0., attn_drop_rate=0., 
                 
                 drop_path_rate=0.,

                 num_groups=256, group_size=32,
                 subsample='fps',  # random, FPS
                 group='ballquery', 
                 radius=0.1,

                 norm_args=None,
                 act_args=None,
                 ):
        super().__init__()
        # grouper
        self.group_divider = SubsampleGroup(num_groups, group_size, subsample, group, radius)
        # define the encoder
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim = embed_dim,
            depth = depth,
            drop_path_rate = dpr,
            num_heads = num_heads, 
            act_args=act_args, norm_args=norm_args
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        ckpt = ckpt['base_model'] if hasattr(ckpt, 'base_model') else ckpt['model']
        
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            logging.info('missing_keys')
            logging.info(
                get_missing_parameters_message(incompatible.missing_keys),
            )
        if incompatible.unexpected_keys:
            logging.info('unexpected_keys')
            logging.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                
            )
        logging.info(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts):
        # divide the point cloud in the same form. 
        neighborhood, center = self.group_divider(pts)
        
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret
