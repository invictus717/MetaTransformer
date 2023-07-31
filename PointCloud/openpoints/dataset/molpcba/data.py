import numpy as np
import torch
from ..dataset_base import DatasetBase
from ..graph_dataset import GraphDataset
from ..graph_dataset import SVDEncodingsGraphDataset
from ..graph_dataset import StructuralDataset

class MOLPCBADataset(DatasetBase):
    def __init__(self, 
                 dataset_path             ,
                 dataset_name = 'MOLPCBA' ,
                 **kwargs
                 ):
        super().__init__(dataset_name = dataset_name,
                         **kwargs)
        self.dataset_path    = dataset_path
    
    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            from ogb.graphproppred import GraphPropPredDataset
            self._dataset = GraphPropPredDataset(name='ogbg-molpcba', root=self.dataset_path)
            return self._dataset

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training':'train', 
                     'validation':'valid', 
                     'test':'test'}[self.split]
            self._record_tokens = self.dataset.get_idx_split()[split]
            return self._record_tokens
    
    def read_record(self, token):
        graph, target = self.dataset[token]
        graph['num_nodes'] = np.array(graph['num_nodes'], dtype=np.int16)
        graph['edges'] = graph.pop('edge_index').T.astype(np.int16)
        graph['edge_features'] = graph.pop('edge_feat').astype(np.int16)
        graph['node_features'] = graph.pop('node_feat').astype(np.int16)
        graph['target'] = np.array(target, np.float32)
        return graph



class MOLPCBAGraphDataset(GraphDataset,MOLPCBADataset):
    pass

class MOLPCBASVDGraphDataset(SVDEncodingsGraphDataset,MOLPCBADataset):
    pass

class MOLPCBAStructuralGraphDataset(StructuralDataset,MOLPCBAGraphDataset):
    pass

class MOLPCBAStructuralSVDGraphDataset(StructuralDataset,MOLPCBASVDGraphDataset):
    pass
