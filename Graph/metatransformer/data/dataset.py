"""
Modified from https://github.com/microsoft/Graphormer
"""

from functools import lru_cache

import numpy as np
import torch
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset

from .wrapper import preprocess_item
from .collator import collator

from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from .ogb_datasets import OGBDatasetLookupTable


class BatchedDataDataset(FairseqDataset):
    def __init__(
            self,
            dataset,
            max_node=128,
            multi_hop_max_dist=5,
            spatial_pos_max=1024
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return preprocess_item(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_node
        )


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


class TokenGTDataset:
    def __init__(
            self,
            dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
            dataset_spec: Optional[str] = None,
            dataset_source: Optional[str] = None,
            seed: int = 0,
            train_idx=None,
            valid_idx=None,
            test_idx=None,
    ):
        super().__init__()
        if dataset is not None:
            raise ValueError("customized dataset can only have source pyg or dgl")
        if dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed=seed)

        self.setup()
        list_N = []
        for i in range(len(self.dataset_val)):
            list_N.append(self.dataset_val[i].x.shape[0])
        for i in range(len(self.dataset_test)):
            list_N.append(self.dataset_test[i].x.shape[0])
        list_N = torch.tensor(list_N)
        torch.save(list_N, "val_test_list_N.pt")

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
