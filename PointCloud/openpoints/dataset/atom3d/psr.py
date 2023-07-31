import os
import numpy as np
from atom3d.datasets import LMDBDataset
from ..build import DATASETS
from ...transforms import  DataTransforms

prot_atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'CL', 'SE', 'F']


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


@DataTransforms.register_module()
class Atom2Points(object):
    def __call__(self, item):
        # Transform protein into voxel grids.
        # Apply random rotation matrix.
        id = eval(item['id'])
        transformed = {
            'pos': item['atoms'][['x', 'y', 'z']].to_numpy().astype(np.float32),
            'features': np.array([one_of_k_encoding_unk(e, prot_atoms) for e in item['atoms']['element']]).astype(np.float32).transpose(1, 0),
            'label': np.float32(item['scores']['gdt_ts']),
            'target': id[0],
            'decoy': id[1],
        }
        return transformed


@DATASETS.register_module()
class AtomPSR(LMDBDataset):
    def __init__(self, data_dir, split, transform=Atom2Points()):
        assert split in ['train', 'val', 'test']
        super().__init__(os.path.join(data_dir, split), transform)

