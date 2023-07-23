# Copyright (c) Shanghai AI Lab. All rights reserved.
from .beit_adapter import BEiTAdapter
from .uniperceiver_adapter import UniPerceiverAdapter
from .vit_adapter import ViTAdapter
from .vit_baseline import ViTBaseline

__all__ = ['UniPerceiverAdapter', 'ViTAdapter', 'ViTBaseline', 'BEiTAdapter']
