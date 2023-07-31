import torch
import torch.nn as nn
from .simpleview_util import PCViews
from ..build import MODELS
# from roi_align import CropAndResize # crop_and_resize module
from openpoints.transforms import build_transforms_from_cfg


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.squeeze()


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)


@MODELS.register_module()
class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
            BatchNormPoint(in_features),
            # dropout before concatenation so that each view drops features independently
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(in_features=in_features * self.num_views,
                      out_features=in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_features, out_features=out_features,
                      bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        out = self.model(feat)
        return out


@MODELS.register_module()
class MVModel(nn.Module):
    def __init__(self, task='cls', backbone='resnet18',
                 channels=16,
                 num_classes=15,
                 resolution=128,
                 use_img_transform=False,
                 **kwargs):
        super().__init__()
        assert task == 'cls'
        self.task = task
        self.num_classes = num_classes
        self.dropout = kwargs.get('dropout', 0.5)
        self.channels = channels
        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img

        img_layers, in_features = self.get_img_layers(
            backbone, channels=channels)
        self.img_model = nn.Sequential(*img_layers)

        self.final_fc = MVFC(
            num_views=self.num_views,
            in_features=in_features,
            out_features=self.num_classes,
            dropout=self.dropout)
        if use_img_transform:
            self.img_transform = build_transforms_from_cfg('img', {'img': ['Zoom']})
        else:
            self.img_transform = None
            
    def forward(self, pc):
        """
        :param pc:
        :return:
        """
        if hasattr(pc, 'keys'):
            pc = pc['pos']
        img = self.get_img(pc)

        if self.training and self.img_transform is not None:
            img = self.img_transform(img, pc.shape[0] * self.num_views)
        feat = self.img_model(img)
        logit = self.final_fc(feat)
        return logit

    def forward_cls_feat(self, pc):
        return self.forward(pc)

    def get_img(self, pc):
        img = self._get_img(pc)
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)
        return img

    @staticmethod
    def get_img_layers(backbone, channels):
        """
        Return layers for the image model
        """

        from .resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=channels,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features
