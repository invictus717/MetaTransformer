# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import random, logging
import numpy as np
import torch
import collections
from .transforms_factory import DataTransforms
#
# import scipy
# import scipy.ndimage
# import scipy.interpolate
from scipy.linalg import expm, norm


@DataTransforms.register_module()
class PointCloudToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        pts = data['pos']
        normals = data['normals'] if 'normals' in data.keys() else None
        colors = data['colors'] if 'colors' in data.keys() else None
        data['pos'] = torch.from_numpy(pts).float()
        if normals is not None:
            data['normals'] = torch.from_numpy(normals).float().transpose(0, 1)
        if colors is not None:
            data['colors'] = torch.from_numpy(colors).transpose(0, 1).float()
        return data


@DataTransforms.register_module()
class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - torch.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim + 1]
                data['heights'] = height - torch.min(height)

            if self.centering:
                data['pos'] = data['pos'] - torch.mean(data['pos'], axis=0, keepdims=True)

            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - torch.mean(data, axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data = data / m
        return data


@DataTransforms.register_module()
class PointCloudXYZAlign(object):
    """Centering the point cloud in the xy plane
    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 gravity_dim=2,
                 **kwargs):
        self.gravity_dim = gravity_dim

    def __call__(self, data):
        if hasattr(data, 'keys'):
            data['pos'] -= torch.mean(data['pos'], axis=0, keepdims=True)
            data['pos'][:, self.gravity_dim] -= torch.min(data['pos'][:, self.gravity_dim])
        else:
            data -= torch.mean(data, axis=0, keepdims=True)
            data[:, self.gravity_dim] -= torch.min(data[:, self.gravity_dim])
        return data


@DataTransforms.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.2, **kwargs):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data):
        if random.random() < self.dropout_application_ratio:
            N = len(data['pos'])
            inds = torch.randperm(N)[:int(N * (1 - self.dropout_ratio))]
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v[inds]
        return data


@DataTransforms.register_module()
class RandomHorizontalFlip(object):
    def __init__(self, upright_axis, aug_prob=0.95, **kwargs):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.D = 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
        self.aug_prob = aug_prob

    def __call__(self, data):
        if random.random() < self.aug_prob:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(data['pos'])
                    data['pos'][:, curr_ax] = coord_max - data['pos'][:, curr_ax]
                    if 'normals' in data:
                        data['normals'][:, curr_ax] = -data['normals'][:, curr_ax]

        return data


@DataTransforms.register_module()
class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


@DataTransforms.register_module()
class PointCloudTranslation(object):
    def __init__(self, shift=[0.2, 0.2, 0.], **kwargs):
        self.shift = torch.from_numpy(np.array(shift)).to(torch.float32)

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        translation = torch.rand(3, dtype=torch.float32, device=device) * self.shift.to(device)
        if hasattr(data, 'keys'):
            data['pos'] += translation
        else:
            data += translation
        return data


@DataTransforms.register_module()
class PointCloudScaleAndTranslate(object):
    def __init__(self, scale=[2. / 3, 3. / 2], scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 anisotropic=True,
                 shift=[0.2, 0.2, 0.2],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.shift = torch.from_numpy(np.array(shift)).to(torch.float32)
        self.scale_xyz = scale_xyz
        self.anisotropic = anisotropic
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(self.mirror>0) != 0

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        # * note : scale_xyz has higher priority than mirror
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        translation = (torch.rand(3, dtype=torch.float32, device=device) - 0.5) * 2 * self.shift.to(device)
        if hasattr(data, 'keys'):
            data['pos'] = torch.mul(data['pos'], scale) + translation
        else:
            data = torch.mul(data, scale) + translation
        return data


@DataTransforms.register_module()
class PointCloudJitter(object):
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        if hasattr(data, 'keys'):
            noise = torch.randn_like(data['pos']) * self.noise_std
            data['pos'] += noise.clamp_(-self.noise_clip, self.noise_clip)
        else:
            noise = torch.randn_like(data) * self.noise_std
            data += noise.clamp_(-self.noise_clip, self.noise_clip)
        return data


@DataTransforms.register_module()
class PointCloudScaleAndJitter(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 scale_xyz=[True, True, True],  # ratio for xyz dimenions
                 anisotropic=True,  # scaling in different ratios for x, y, z
                 jitter_sigma=0.01, jitter_clip=0.05,
                 mirror=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.scale_xyz = scale_xyz
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip
        self.anisotropic = anisotropic
        self.mirror = torch.from_numpy(np.array(mirror))

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        mirror = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.mirror = self.mirror.to(device)
        mirror = mirror * self.mirror + (1 - self.mirror)
        scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            noise = (torch.randn_like(data['pos']) * self.noise_std).clamp_(-self.noise_clip, self.noise_clip)
            data['pos'] = torch.mul(data['pos'], scale) + noise
        else:
            noise = (torch.randn_like(data) * self.noise_std).clamp_(-self.noise_clip, self.noise_clip)
            data = torch.mul(data, scale) + noise
        return data


@DataTransforms.register_module()
class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data


# @DataTransforms.register_module()
# class ChromaticTranslation(object):
#     """Add random color to the image, input must be an array in [0,255] or a PIL image"""
#
#     def __init__(self, trans_range_ratio=1e-1, aug_prob=0.95, **kwargs):
#         """
#         trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
#         """
#         self.trans_range_ratio = trans_range_ratio
#         self.aug_prob = aug_prob
#
#     def __call__(self, data):
#         if 'colors' in data:
#             if random.random() < self.aug_prob:
#                 tr = (torch.rand(1, 3, device=data['colors'].device) - 0.5) * 255 * 2 * self.trans_range_ratio
#                 data['colors'] = torch.clamp(tr + data['colors'], 0, 255)
#         return data
#

#
# @DataTransforms.register_module()
# class PointCloudRandomScale(object):#
# # @DataTransforms.register_module()
# # class ChromaticAutoContrast(object):
# #     def __init__(self, randomize_blend_factor=True, blend_factor=0.5, **kwargs):
# #         self.randomize_blend_factor = randomize_blend_factor
# #         self.blend_factor = blend_factor
# #
# #     def __call__(self, data, **kwargs):
# #         if 'colors' in data:
# #             if random.random() < 0.2:
# #                 # to avoid chromatic drop problems
# #                 if data['colors'].mean() <= 0.1:
# #                     return data
# #                 lo = data['colors'].min(1, keepdims=True)[0]
# #                 hi = data['colors'].max(1, keepdims=True)[0]
# #                 scale = 255 / (hi - lo)
# #                 contrast_feats = (data['colors'] - lo) * scale
# #                 blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
# #                 data['colors'] = (1 - blend_factor) * data['colors'] + blend_factor * contrast_feats
# #         return data
# #
# #
# # @DataTransforms.register_module()
# # class ChromaticJitter(object):
# #
# #     def __init__(self, std=0.01, **kwargs):
# #         self.std = std
# #
# #     def __call__(self, data):
# #         if 'colors' in data:
# #             if random.random() < 0.95:
# #                 noise = torch.randn_like(data['colors'])
# #                 noise *= self.std * 255
# #                 data['colors'] = torch.clamp(noise + data['colors'], 0, 255)
# #         return data


@DataTransforms.register_module()
class ChromaticDropGPU(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.color_drop = color_drop

    def __call__(self, data):
        if torch.rand(1) < self.color_drop:
            data['x'][:, :3] = 0
        return data


@DataTransforms.register_module()
class ChromaticPerDropGPU(object):
    def __init__(self, color_drop=0.2, **kwargs):
        self.color_drop = color_drop

    def __call__(self, data):
        colors_drop = (torch.rand((data['x'].shape[0], 1)) > self.color_drop).to(torch.float32)
        data['x'][:, :3] *= colors_drop
        return data


@DataTransforms.register_module()
class ChromaticNormalize(object):
    def __init__(self,
                 color_mean=[0.5136457, 0.49523646, 0.44921124],
                 color_std=[0.18308958, 0.18415008, 0.19252081],
                 **kwargs):
        self.color_mean = torch.from_numpy(np.array(color_mean)).to(torch.float32)
        self.color_std = torch.from_numpy(np.array(color_std)).to(torch.float32)

    def __call__(self, data):
        device = data['x'].device
        if data['x'][:, :3].max() > 1:
            data['x'][:, :3] /= 255.
        data['x'][:, :3] = (data['x'][:, :3] - self.color_mean.to(device)) / self.color_std.to(device)
        return data


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


class Cutmix:
    """ Cutmix that applies different params to each element or whole batch
    Update: 1. random cutmix does not work on classification (ScanObjectNN, PointNext), April 7, 2022
    Args:
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        prob (float): probability of applying mixup or cutmix per batch or element
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, cutmix_alpha=0.3, prob=1.0,
                 label_smoothing=0.1, num_classes=1000):
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _mix_batch(self, data):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        # the trianing batches should have same size. 
        if hasattr(data, 'keys'):  # data is a dict
            # pos, feat? 
            N = data['pos'].shape[1]
            n_mix = int(N * lam)
            data['pos'][:, -n_mix:] = data['pos'].flip(0)[:, -n_mix:]

            if 'x' in data.keys():
                data['x'][:, :, -n_mix:] = data['x'].flip(0)[:, :, -n_mix:]
        else:
            data[:, -n_mix:] = data.flip(0)[:, -n_mix:]
        return lam

    def __call__(self, data, target):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        lam = self._mix_batch(data)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device)
        return data, target


# from roi_align import CropAndResize  # crop_and_resize module
# from math import sqrt
#
#
# @DataTransforms.register_module()
# class Zoom(object):
#     def __init__(self, ratio=0.35, IMG_MEAN=0.044471418750000005, IMG_VAR=0.061778141, resolution=128, views=6,
#                  **kwargs):  # simple view hyper params
#         # super().__init__()
#         self.ratio = ratio
#         self.extrapolation_value = ((0 - IMG_MEAN) / sqrt(IMG_VAR))
#         self.resolution = resolution
#         self.CropAndResize = CropAndResize(resolution, resolution, self.extrapolation_value)
#         # self.batch = resolution * views
#
#     def __call__(self, data, batch):
#         if self.extrapolation_value == 0:
#             print("WARNING: using 0 for the extrapolated value")
#         boxes = torch.cat((torch.zeros((batch, 2), device=data.device, dtype=torch.float32),
#                            torch.ones((batch, 2), device=data.device, dtype=torch.float32)), dim=1)
#         ind = torch.arange(batch, device=data.device, dtype=torch.int)
#         num = (torch.rand((batch, 4), device=data.device) - 0.5) * 2 * self.ratio
#         boxes = torch.add(boxes, num)
#         return self.CropAndResize(
#             data, boxes=boxes, box_ind=ind
#         )


# # # Numpy Operation
# # @DataTransforms.register_module()
# # class RGBtoHSV(object):
# #
# #     def __init__(self, **kwargs):
# #         pass
# #
# #     def __call__(self, data):
# #         if 'colors' in data:
# #             # print("hsv")
# #             # Translated from source of colorsys.rgb_to_hsv
# #             # r,g,b should be a numpy arrays with values between 0 and 255
# #             # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
# #             rgb = data['colors'].cpu().float().numpy()
# #             hsv = np.zeros_like(rgb)
# #
# #             r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
# #             maxc = np.max(rgb[..., :3], axis=-1)
# #             minc = np.min(rgb[..., :3], axis=-1)
# #             hsv[..., 2] = maxc
# #             mask = maxc != minc
# #             hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
# #             rc = np.zeros_like(r)
# #             gc = np.zeros_like(g)
# #             bc = np.zeros_like(b)
# #             rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
# #             gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
# #             bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
# #             hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
# #             hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
# #             data['colors'] = torch.from_numpy(hsv).to(data['colors'].device)
# #         return data
# #
# #
# # # Numpy Operation
# # @DataTransforms.register_module()
# # class HSVtoRGB(object):
# #     def __init__(self, **kwargs):
# #         pass
# #         # Translated from source of colorsys.hsv_to_rgb
# #         # h,s should be a numpy arrays with values between 0.0 and 1.0
# #         # v should be a numpy array with values between 0.0 and 255.0
# #         # hsv_to_rgb returns an array of uints between 0 and 255.
# #
# #     def __call__(self, data):
# #         if 'colors' in data:
# #             hsv = data['colors'].cpu().float().numpy()
# #             rgb = np.empty_like(hsv)
# #             rgb[..., 3:] = hsv[..., 3:]
# #             h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
# #             i = (h * 6.0).astype('uint8')
# #             f = (h * 6.0) - i
# #             p = v * (1.0 - s)
# #             q = v * (1.0 - s * f)
# #             t = v * (1.0 - s * (1.0 - f))
# #             i = i % 6
# #             conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
# #             rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
# #             rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
# #             rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
# #             rgb = rgb.astype('uint8')
# #             data['colors'] = torch.from_numpy(rgb).to(data['colors'].device)
# #         return rgb
# #
# #
# # # POINT TRANSFORMER CHROMATIC JITTER
# # @DataTransforms.register_module()
# # class ChromaticJitters(object):
# #     def __init__(self, p=0.95, std=0.005, **kwargs):
# #         self.p = p
# #         self.std = std
# #
# #     def __call__(self, data):
# #         if np.random.rand() < self.p:
# #             noise = np.random.randn(data['colors'].shape[0], 3)
# #             noise *= self.std * 255
# #             data['colors'][:, :3] = torch.clamp(noise + data['colors'][:, :3], 0, 255)
# #         return data
# #
# #
# # @DataTransforms.register_module()
# # class HueSaturationTranslation(object):
# #     # @staticmethod
# #     def rgb_to_hsv(rgb):
# #         # Translated from source of colorsys.rgb_to_hsv
# #         # r,g,b should be a numpy arrays with values between 0 and 255
# #         # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
# #         # rgb = rgb.astype('float')
# #         hsv = torch.zeros_like(rgb)
# #         # in case an RGBA array was passed, just copy the A channel
# #         hsv[..., 3:] = rgb[..., 3:]
# #         r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
# #         maxc = torch.max(rgb[..., :3], axis=-1)[0]
# #         minc = torch.min(rgb[..., :3], axis=-1)[0]
# #         hsv[..., 2] = maxc
# #         mask = maxc != minc
# #         hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
# #         rc = torch.zeros_like(r)
# #         gc = torch.zeros_like(g)
# #         bc = torch.zeros_like(b)
# #         rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
# #         gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
# #         bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
# #         hsv[..., 0] = torch.where(r == maxc, bc - gc, (torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc)))
# #         # hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
# #         hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
# #         return hsv
# #
# #     # @staticmethod
# #     def hsv_to_rgb(hsv):
# #         # Translated from source of colorsys.hsv_to_rgb
# #         # h,s should be a numpy arrays with values between 0.0 and 1.0
# #         # v should be a numpy array with values between 0.0 and 255.0
# #         # hsv_to_rgb returns an array of uints between 0 and 255.
# #         rgb = torch.empty_like(hsv)
# #         rgb[..., 3:] = hsv[..., 3:]
# #         h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
# #         i = (h * 6.0).type(torch.cuda.ByteTensor)
# #         f = (h * 6.0) - i
# #         p = v * (1.0 - s)
# #         q = v * (1.0 - s * f)
# #         t = v * (1.0 - s * (1.0 - f))
# #         i = i % 6
# #         conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
# #
# #         rgb[..., 0] = torch.where((conditions[0]) | (conditions[5]), v, torch.where(conditions[1], q, torch.where(
# #             (conditions[2]) | (conditions[3]), p, torch.where(conditions[4], t, v))))
# #         rgb[..., 1] = torch.where((conditions[0]) | (conditions[1]) | (conditions[2]), v,
# #                                   torch.where(conditions[3], q, torch.where((conditions[4]) | (conditions[5]), p, t)))
# #         rgb[..., 2] = torch.where((conditions[0]) | (conditions[3]) | (conditions[4]), v,
# #                                   torch.where(conditions[1], p, torch.where(
# #                                       conditions[2], t, torch.where(conditions[5], q, p))))
# #         # rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
# #         # rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
# #         # rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
# #         return rgb.type(torch.cuda.ByteTensor)
# #
# #     def __init__(self, hue_max=0.5, saturation_max=0.2, **kwargs):
# #         self.hue_max = hue_max
# #         self.saturation_max = saturation_max
# #
# #     def __call__(self, data):
# #         # Assume feat[:, :3] is rgb
# #         if 'colors' in data:
# #             rgb = data['colors']
# #             hsv = HueSaturationTranslation.rgb_to_hsv(rgb[:, :3])
# #             hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
# #             sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
# #             hsv[..., 0] = torch.remainder(hue_val + hsv[..., 0] + 1, 1)
# #             hsv[..., 1] = torch.clamp(sat_ratio * hsv[..., 1], 0, 1)
# #             data['colors'][:, :3] = torch.clamp(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
# #         return data
#     def __init__(self, scale=[0.9, 1.1], anisotropic=False, **kwargs):
#         self.scale = scale
#         self.anisotropic = anisotropic
#
#     def __call__(self, data):
#         if hasattr(data, 'keys'):
#             scale = torch.distributions.Uniform(self.scale[0], self.scale[1]).sample((3 if self.anisotropic else 1,)
#                                                                                      ).to(data['pos'].device)
#             data['pos'] *= scale
#         else:
#             scale = torch.distributions.Uniform(self.scale[0], self.scale[1]).sample((3 if self.anisotropic else 1,)
#                                                                                      ).to(data.device)
#             data *= scale
#         return data
#
#
# @DataTransforms.register_module()
# class PointCloudRandomShift(object):
#     def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
#         self.shift = shift
#
#     def __call__(self, data):
#         if hasattr(data, 'keys'):
#             shift_x = torch.distributions.Uniform(-self.shift[0], self.shift[0]).sample((1,)).to(data['pos'].device)
#             shift_y = torch.distributions.Uniform(-self.shift[1], self.shift[1]).sample((1,)).to(data['pos'].device)
#             shift_z = torch.distributions.Uniform(-self.shift[1], self.shift[1]).sample((1,)).to(data['pos'].device)
#             data['pos'] += [shift_x, shift_y, shift_z]
#         else:
#             shift_x = torch.distributions.Uniform(-self.shift[0], self.shift[0]).sample((1,)).to(data.device)
#             shift_y = torch.distributions.Uniform(-self.shift[1], self.shift[1]).sample((1,)).to(data.device)
#             shift_z = torch.distributions.Uniform(-self.shift[1], self.shift[1]).sample((1,)).to(data.device)
#             data += [shift_x, shift_y, shift_z]
#         return data
#
#
# @DataTransforms.register_module()
# class RandomFlip(object):
#     def __init__(self, p=0.5, **kwargs):
#         self.p = p
#
#     def __call__(self, data):
#         if hasattr(data, 'keys'):
#             if np.random.rand() < self.p:
#                 data['pos'][:, 0] = -data['pos'][:, 0]
#             if np.random.rand() < self.p:
#                 data['pos'][:, 1] = -data['pos'][:, 1]
#         else:
#             if np.random.rand() < self.p:
#                 data[:, 0] = -data[:, 0]
#             if np.random.rand() < self.p:
#                 data[:, 1] = -data[:, 1]
#         return data
#
#
# @DataTransforms.register_module()
# class RandomJitter(object):
#     def __init__(self, sigma=0.01, clip=0.05, **kwargs):
#         self.sigma = sigma
#         self.clip = clip
#
#     def __call__(self, data):
#         assert (self.clip > 0)
#         if hasattr(data, 'keys'):
#             jitter = torch.clamp(self.sigma * torch.randn(data['pos'].shape[0], 3), -1 * self.clip, self.clip)
#             data['pos'] += jitter
#         else:
#             jitter = torch.clamp(self.sigma * torch.randn(data.shape[0], 3), -1 * self.clip, self.clip)
#             data += jitter
#         return data
#
#
# @DataTransforms.register_module()
# class RandomDropColor(object):
#     def __init__(self, p=0.2, **kwargs):
#         self.p = p
#
#     def __call__(self, data):
#         if 'colors' in data:
#             if np.random.rand() < self.p:
#                 data['colors'][:, :3] = 0
#         return data

#
# class ElasticDistortion(object):
#     def __init__(self, granularity=0.2, magnitude=0.4, **kwargs):
#         self.granularity = granularity
#         self.magnitude = magnitude
#
#     def __call__(self, data):
#         """Apply elastic distortion on sparse coordinate space.
#
#           pointcloud: numpy array of (number of points, at least 3 spatial dims)
#           granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
#           magnitude: noise multiplier
#         """
#         if random.random() < 0.95:
#             blurx = torch.ones((3, 1, 1, 1), dtype=torch.float32) / 3
#             blury = torch.ones((1, 3, 1, 1), dtype=torch.float32) / 3
#             blurz = torch.ones((1, 1, 3, 1), dtype=torch.float32) / 3
#             coords = data['pos']
#             coords_min = coords.min(0)[0]
#
#             # Create Gaussian noise tensor of the size given by granularity.
#             noise_dim = ((coords - coords_min).max(0)[0] // self.granularity).int() + 3
#             noise = torch.randn(*noise_dim, 3, dtype=torch.float32)
#
#             # Smoothing.
#             for _ in range(2):
#                 noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
#                 noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
#                 noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)
#
#             # Trilinear interpolate noise filters for each spatial dimensions.
#             ax = [
#                 torch.linspace(d_min, d_max, d)
#                 for d_min, d_max, d in zip(coords_min - self.granularity, coords_min + self.granularity *
#                                            (noise_dim - 2), noise_dim)
#             ]
#             interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
#             data['pos'] = coords + interp(coords) * self.magnitude
#         return data
