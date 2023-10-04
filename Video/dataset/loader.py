import io

import cv2
import numpy as np
from decord import VideoReader, cpu

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False


def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(video_path):
        if _client is not None and 's3:' in video_path:
            video_path = io.BytesIO(_client.get(video_path))

        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr

    return _loader


def get_image_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(frame_path):
        if _client is not None and 's3:' in frame_path:
            img_bytes = _client.get(frame_path)
        else:
            with open(frame_path, 'rb') as f:
                img_bytes = f.read()

        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img

    return _loader
