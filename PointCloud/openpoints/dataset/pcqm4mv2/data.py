"""
https://github.com/shamim-hussain/egt_pytorch
"""

import numpy as np
from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset
from ..build import DATASETS


@DATASETS.register_module()
class PCQM4Mv2Dataset(DatasetBase):
    def __init__(self,
                 dataset_path,
                 dataset_name='PCQM4MV2',
                 **kwargs
                 ):
        super().__init__(dataset_name=dataset_name,
                         **kwargs)
        self.dataset_path = dataset_path

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            from ogb.lsc import PCQM4Mv2Dataset
            # smiles: simplified molecular-input line-entry system (SMILES). sequence representation of molecules
            # The molecules are provided as SMILES strings (sequence representation of molecules)
            # i = 1234
            # print(dataset[i]) # ('CC(NCC[C@H]([C@@H]1CCC(=CC1)C)C)C', 6.811009678015001)

            # The second option provides a molecular graph object constructed from the SMILES string.
            # After preprocessing, the file size will be around 8GB.
            # # get i-th molecule and its target value (nan for test data)
            # i = 1234
            # print(dataset[i]) # (graph_obj, 6.811009678015001)

            # details: https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
            from ogb.utils import smiles2graph
            self._smiles2graph = smiles2graph
            self._dataset = PCQM4Mv2Dataset(root=self.dataset_path, only_smiles=True)
            return self._dataset

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training': 'train',
                     'validation': 'valid',
                     'test': 'test-dev',
                     'challenge': 'test-challenge'}[self.split]
            self._record_tokens = self.dataset.get_idx_split()[split]
            return self._record_tokens

    def read_record(self, token):
        smiles, target = self.dataset[token]
        graph = self._smiles2graph(smiles)
        graph['num_nodes'] = np.array(graph['num_nodes'], dtype=np.int16)
        graph['edges'] = graph.pop('edge_index').T.astype(np.int16)
        graph['edge_features'] = graph.pop('edge_feat').astype(np.int16)
        graph['node_features'] = graph.pop('node_feat').astype(np.int16)
        graph['target'] = np.array(target, np.float32)
        return graph


@DATASETS.register_module()
class PCQM4Mv2GraphDataset(GraphDataset, PCQM4Mv2Dataset):
    pass


@DATASETS.register_module()
class PCQM4Mv2SVDGraphDataset(SVDEncodingsGraphDataset, PCQM4Mv2Dataset):
    pass


@DATASETS.register_module()
class PCQM4Mv2StructuralGraphDataset(StructuralDataset, PCQM4Mv2GraphDataset):
    pass


@DATASETS.register_module()
class PCQM4Mv2StructuralSVDGraphDataset(StructuralDataset, PCQM4Mv2SVDGraphDataset):
    pass
