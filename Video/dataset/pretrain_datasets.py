import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .loader import get_image_loader, get_video_loader
from .masking_generator import (
    RunningCellMaskingGenerator,
    TubeMaskingGenerator,
)
from .transforms import (
    GroupMultiScaleCrop,
    GroupNormalize,
    Stack,
    ToTorchFormatTensor,
)


class DataAugmentationForVideoMAEv2(object):

    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        div = True
        roll = False
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size,
                                                      [1, .875, .75, .66])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=roll),
            ToTorchFormatTensor(div=div),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.encoder_mask_map_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio)
        else:
            raise NotImplementedError(
                'Unsupported encoder masking strategy type.')
        if args.decoder_mask_ratio > 0.:
            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(
                    args.window_size, args.decoder_mask_ratio)
            else:
                raise NotImplementedError(
                    'Unsupported decoder masking strategy type.')

    def __call__(self, images):
        process_data, _ = self.transform(images)
        encoder_mask_map = self.encoder_mask_map_generator()
        if hasattr(self, 'decoder_mask_map_generator'):
            decoder_mask_map = self.decoder_mask_map_generator()
        else:
            decoder_mask_map = 1 - encoder_mask_map
        return process_data, encoder_mask_map, decoder_mask_map

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAEv2,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Encoder Masking Generator = %s,\n" % str(
            self.encoder_mask_map_generator)
        if hasattr(self, 'decoder_mask_map_generator'):
            repr += "  Decoder Masking Generator = %s,\n" % str(
                self.decoder_mask_map_generator)
        else:
            repr += "  Do not use decoder masking,\n"
        repr += ")"
        return repr


class HybridVideoMAE(torch.utils.data.Dataset):
    """Load your own videomae pretraining dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are four items in each line:
        (1) video path; (2) start_idx, (3) total frames and (4) video label.
        for pre-train video data
            total frames < 0, start_idx and video label meaningless
        for pre-train rawframe data
            video label meaningless
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default 'img_{:05}.jpg'.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    num_sample : int, default 1.
        Number of sampled views for Repeated Augmentation.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 lazy_init=False,
                 num_sample=1):

        super(HybridVideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        # NOTE:
        # for hybrid train
        # different frame naming formats are used for different datasets
        # should MODIFY the fname_tmpl to your own situation
        self.ava_fname_tmpl = 'image_{:06}.jpg'
        self.ssv2_fname_tmpl = 'img_{:05}.jpg'

        # NOTE:
        # we set sampling_rate = 2 for ssv2
        # thus being consistent with the fine-tuning stage
        # Note that the ssv2 we use is decoded to frames at 12 fps;
        # if decoded at 24 fps, the sample interval should be 4.
        self.ssv2_skip_length = self.new_length * 2
        self.orig_skip_length = self.skip_length

        self.video_loader = get_video_loader()
        self.image_loader = get_image_loader()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            video_name, start_idx, total_frame = self.clips[index]
            self.skip_length = self.orig_skip_length

            if total_frame < 0:
                decord_vr = self.video_loader(video_name)
                duration = len(decord_vr)

                segment_indices, skip_offsets = self._sample_train_indices(
                    duration)
                frame_id_list = self.get_frame_id_list(duration,
                                                       segment_indices,
                                                       skip_offsets)
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                images = [
                    Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
                    for vid, _ in enumerate(frame_id_list)
                ]

            else:
                # ssv2 & ava & other rawframe dataset
                if 'SomethingV2' in video_name:
                    self.skip_length = self.ssv2_skip_length
                    fname_tmpl = self.ssv2_fname_tmpl
                elif 'AVA2.2' in video_name:
                    fname_tmpl = self.ava_fname_tmpl
                else:
                    fname_tmpl = self.name_pattern

                segment_indices, skip_offsets = self._sample_train_indices(
                    total_frame)
                frame_id_list = self.get_frame_id_list(total_frame,
                                                       segment_indices,
                                                       skip_offsets)

                images = []
                for idx in frame_id_list:
                    frame_fname = os.path.join(
                        video_name, fname_tmpl.format(idx + start_idx))
                    img = self.image_loader(frame_fname)
                    img = Image.fromarray(img)
                    images.append(img)

        except Exception as e:
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

        if self.num_sample > 1:
            process_data_list = []
            encoder_mask_list = []
            decoder_mask_list = []
            for _ in range(self.num_sample):
                process_data, encoder_mask, decoder_mask = self.transform(
                    (images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
            return process_data_list, encoder_mask_list, decoder_mask_list
        else:
            process_data, encoder_mask, decoder_mask = self.transform(
                (images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, encoder_mask, decoder_mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                item = (clip_path, start_idx, total_frame)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


class VideoMAE(torch.utils.data.Dataset):
    """Load your own videomae pretraining dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are four items in each line:
        (1) video path; (2) start_idx, (3) total frames and (4) video label.
        for pre-train video data
            total frames < 0, start_idx and video label meaningless
        for pre-train rawframe data
            video label meaningless
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default 'img_{:05}.jpg'.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    num_sample : int, default 1.
        Number of sampled views for Repeated Augmentation.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 lazy_init=False,
                 num_sample=1):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.video_loader = get_video_loader()
        self.image_loader = get_image_loader()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            video_name, start_idx, total_frame = self.clips[index]
            if total_frame < 0:  # load video
                decord_vr = self.video_loader(video_name)
                duration = len(decord_vr)

                segment_indices, skip_offsets = self._sample_train_indices(
                    duration)
                frame_id_list = self.get_frame_id_list(duration,
                                                       segment_indices,
                                                       skip_offsets)
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                images = [
                    Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
                    for vid, _ in enumerate(frame_id_list)
                ]
            else:  # load frames
                segment_indices, skip_offsets = self._sample_train_indices(
                    total_frame)
                frame_id_list = self.get_frame_id_list(total_frame,
                                                       segment_indices,
                                                       skip_offsets)

                images = []
                for idx in frame_id_list:
                    frame_fname = os.path.join(
                        video_name, self.name_pattern.format(idx + start_idx))
                    img = self.image_loader(frame_fname)
                    img = Image.fromarray(img)
                    images.append(img)

        except Exception as e:
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

        if self.num_sample > 1:
            process_data_list = []
            encoder_mask_list = []
            decoder_mask_list = []
            for _ in range(self.num_sample):
                process_data, encoder_mask, decoder_mask = self.transform(
                    (images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
            return process_data_list, encoder_mask_list, decoder_mask_list
        else:
            process_data, encoder_mask, decoder_mask = self.transform(
                (images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, encoder_mask, decoder_mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, start_idx, total_frames
                if len(line_info) < 3:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                item = (clip_path, start_idx, total_frame)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list
