
import torch
import numpy as np

from ..dataset_base import DatasetBase

from .stack_with_pad import stack_with_pad
from collections import defaultdict
from numba.typed import List


class GraphDataset(DatasetBase):
    def __init__(self,
                 num_nodes_key     = 'num_nodes',
                 edges_key         = 'edges',
                 node_features_key = 'node_features',
                 edge_features_key = 'edge_features',
                 node_mask_key     = 'node_mask',
                 targets_key       = 'target',
                 include_node_mask = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_nodes_key     = num_nodes_key    
        self.edges_key         = edges_key        
        self.node_features_key = node_features_key
        self.edge_features_key = edge_features_key
        self.node_mask_key     = node_mask_key    
        self.targets_key       = targets_key      
        self.include_node_mask = include_node_mask
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.include_node_mask:
            item = item.copy()
            item[self.node_mask_key] = np.ones((item[self.num_nodes_key],), dtype=np.uint8)
        return item
    
    def _calculate_max_nodes(self):
        self._max_nodes = self[0][self.num_nodes_key]
        self._max_nodes_index = 0
        for i in range(1, super().__len__()):
            graph = super().__getitem__(i)
            cur_nodes = graph[self.num_nodes_key]
            if cur_nodes > self._max_nodes:
                self._max_nodes = cur_nodes
                self._max_nodes_index = i
    
    @property
    def max_nodes(self):
        try:
            return self._max_nodes
        except AttributeError:
            self._calculate_max_nodes()
            return self._max_nodes
    
    @property
    def max_nodes_index(self):
        try:
            return self._max_nodes_index
        except AttributeError:
            self._calculate_max_nodes()
            return self._max_nodes_index
    
    def cache_load_and_save(self, base_path, op, verbose):
        super().cache_load_and_save(base_path, op, verbose)
        max_nodes_path = base_path/'max_nodes_data.pt'
        
        if op == 'load':
            max_nodes_data = torch.load(str(max_nodes_path))
            self._max_nodes = max_nodes_data['max_nodes']
            self._max_nodes_index = max_nodes_data['max_nodes_index']
        elif op == 'save':
            if verbose: print(f'Calculating {self.split} max nodes...',flush=True)
            max_nodes_data = {'max_nodes': self.max_nodes,
                              'max_nodes_index': self.max_nodes_index}
            torch.save(max_nodes_data, str(max_nodes_path))
        else:
            raise ValueError(f'Unknown operation: {op}')
    
    def max_batch(self, batch_size, collate_fn):
        return collate_fn([self.__getitem__(self.max_nodes_index)] * batch_size)



def graphdata_collate(batch):
    batch_data = defaultdict(List)
    for elem in batch:
        for k,v in elem.items():
            batch_data[k].append(v)
    
    out = {k:torch.from_numpy(stack_with_pad(dat)) 
                    for k, dat in batch_data.items()}
    return out
