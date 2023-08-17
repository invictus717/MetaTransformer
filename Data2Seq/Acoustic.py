from torch import nn
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, fstride=10, tstride=10):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))

    def forward(self, x):
        # x = (batch_size, time_frame_num, frequency_bins), e.g., (1, 1024, 128)
        # x = x.unsequeeze(1)
        # x = x.transpose(2, 3)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
