# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np


class Cell():

    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class RandomMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames * self.height * self.width  # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196*8]


class TubeMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        return mask  # [196*8]


class RunningCellMaskingGenerator:

    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio

        num_masks_per_cell = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map.flatten())
            mask = np.stack(mask, axis=0)
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Running Cell Masking with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self):
        mask = self.all_mask_maps[np.random.randint(self.cell_size)]
        return np.copy(mask)
