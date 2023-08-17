import torch
from torch import nn
from einops import repeat


class PatchEmbed(nn.Module):
    def __init__(self, near_band=1, img_size=224, patch_size=16, embed_dim=768, dropout=0):
        super().__init__()

        patch_dim = img_size ** 2 * near_band
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_size + 1, embed_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # b (batch), c, h, w -> patch_size
        # Assure that x is like [b, c, (h w)]
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_tokens, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), embed_dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout
        return x




