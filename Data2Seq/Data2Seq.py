import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import clip
import torchaudio
import Hyper_Spectrum
import Text
import Time_Series
import Image
import Video
import Acoustic
import Graph
from transformers.models.clip import CLIPTokenizer
from Text import zero_padding

class Data2Seq(nn.Module):

    def __init__(self,modality,dim):
        super().__init__()
        self.modality = modality
        self.embed_dim = dim
        if self.modality == 'image' or self.modality == 'infrared' or self.modality == 'x-ray':
            self.embed = Image.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'text':
            self.embed = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        elif self.modality == 'video':
            self.embed = Video.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'graph':
            self.embed = Graph.GraphFeatureTokenizer(rand_node_id_dim = self.embed_dim, orf_node_id_dim = self.embed_dim)
        elif self.modality == 'hyper':
            self.embed =  Hyper_Spectrum.PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'time-series' or self.modality == 'imu':
            self.embed =  Time_Series.DataEmbedding(cin = 1, d_model = self.embed_dim)

    def get_audio_embeddings(audio):
        waveform1, sr = torchaudio.load(audio)

        waveform1 = waveform1 - waveform1.mean()

        audio_embedding = torchaudio.compliance.kaldi.fbank(waveform1, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        return audio_embedding

    def forward(self,data):
        if self.modality in ['image', 'infrared', 'x-ray', 'video', 'graph', 'hyper', 'time-series', 'imu','text' ]:
            embeddings = self.embed(data)
        elif self.modality =='text':
            embeddings = self.embed(data)
            embeddings = zero_padding(text_tensor=embeddings, tar_dim = self.embed_dim)
        elif self.modality =='audio':
            embeddings = self.get_audio_embeddings(data)
        return embeddings