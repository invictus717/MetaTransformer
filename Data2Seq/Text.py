import torch
from torch import nn
import clip


def get_text_embeddings(text, tar_dim=768):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    text_tensor = clip.tokenize(text)
    encoding = model.encode_text(text_tensor)
    encoding = zero_padding(encoding, tar_dim, device)
    return encoding


def zero_padding(text_tensor, tar_dim, device=None):
    padding_size = tar_dim - text_tensor.shape[1]
    zero_tensor = torch.zeros((text_tensor.shape[0], padding_size), device=device)
    padded_tensor = torch.cat([text_tensor, zero_tensor], dim=1)
    return padded_tensor
