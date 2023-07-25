"""
Based on Graphormer codebase https://github.com/microsoft/Graphormer
"""

import logging

import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
import importlib

import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    TokenGTDataset,
    EpochShuffleDataset,
)

from ..data import DATASET_REGISTRY
import sys
import os

logger = logging.getLogger(__name__)


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    dataset_name: str = field(
        default="large-scale-regression",
        metadata={"help": "name of the dataset"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or large-scale-regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    dataset_source: str = field(
        default="pyg",
        metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    seed: int = II("common.seed")

    pretrained_model_name: str = field(
        default="none",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    train_epoch_shuffle: bool = field(
        default=True,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )

    user_data_dir: str = field(
        default="",
        metadata={"help": "path to the module of user-defined dataset"},
    )
    

@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """
    Graph prediction (classification or large-scale-regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.user_data_dir != "":
            self.__import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = TokenGTDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    seed=cfg.seed
                )
            else:
                raise ValueError(
                    f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}"
                )
        else:
            self.dm = TokenGTDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                seed=cfg.seed
            )

    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                    not file.startswith("_")
                    and not file.startswith(".")
                    and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test
        else:
            raise ValueError

        batched_data = BatchedDataDataset(
            batched_data,
            max_node=self.max_nodes(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max
        )

        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(dataset),
                seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)

        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None
