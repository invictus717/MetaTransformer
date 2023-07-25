"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Optional
# from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from .pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from torch_geometric.data import Dataset
from ..pyg_datasets import TokenGTPYGDataset
import torch.distributed as dist
import os


class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyPygPCQM4Mv2Dataset, self).process()
        if dist.is_initialized():
            dist.barrier()


class OGBDatasetLookupTable:
    @staticmethod
    def GetOGBDataset(dataset_name: str, seed: int) -> Optional[Dataset]:
        inner_dataset = None
        train_idx = None
        valid_idx = None
        test_idx = None
        if dataset_name == "pcqm4mv2":
            os.system("mkdir -p dataset/pcqm4m-v2/")
            os.system("touch dataset/pcqm4m-v2/RELEASE_v1.txt")
            inner_dataset = MyPygPCQM4Mv2Dataset()
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")
        return (
            None
            if inner_dataset is None
            else TokenGTPYGDataset(
                inner_dataset, seed, train_idx, valid_idx, test_idx
            )
        )
