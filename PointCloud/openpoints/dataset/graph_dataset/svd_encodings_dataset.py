
import numpy as np
import torch
from tqdm import trange
import numba as nb

from .graph_dataset import GraphDataset

class SVDEncodingsDatasetBase:
    def __init__(self,
                 svd_encodings_key = 'svd_encodings',
                 calculated_dim    = 8,
                 output_dim        = 8,
                 random_neg_splits = ['training'],
                 **kwargs):
        if output_dim > calculated_dim:
            raise ValueError('SVD: output_dim > calculated_dim')
        super().__init__(**kwargs)   
        self.svd_encodings_key = svd_encodings_key
        self.calculated_dim    = calculated_dim    
        self.output_dim        = output_dim        
        self.random_neg_splits = random_neg_splits
    
    def calculate_encodings(self, item):
        raise NotImplementedError('SVDEncodingsDatasetBase.calculate_encodings()')
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        token  = self.record_tokens[index]
        
        try:
            encodings = self._svd_encodings[token]
        except AttributeError:
            encodings = self.calculate_encodings(item)
            self._svd_encodings = {token:encodings}
        except KeyError:
            encodings = self.calculate_encodings(item)
            self._svd_encodings[token] = encodings
        
        if self.output_dim < self.calculated_dim:
            encodings = encodings[:,:self.output_dim,:]
        
        if self.split in self.random_neg_splits:
            rn_factors = np.random.randint(0, high=2, size=(encodings.shape[1],1))*2-1 #size=(encodings.shape[0],1,1)
            encodings = encodings * rn_factors.astype(encodings.dtype)
        
        item[self.svd_encodings_key] = encodings.reshape(encodings.shape[0],-1)
        return item
    
    def calculate_all_svd_encodings(self,verbose=1):
        self._svd_encodings = {}
        if verbose:
            print(f'Calculating all {self.split} SVD encodings...', flush=True)
            for index in trange(super().__len__()):
                item = super().__getitem__(index)
                token  = self.record_tokens[index]
                self._svd_encodings[token] = self.calculate_encodings(item)
        else:
            for index in range(super().__len__()):
                item = super().__getitem__(index)
                token = self.record_tokens[index]
                self._svd_encodings[token] = self.calculate_encodings(item)
    
    def cache_load_and_save(self, base_path, op, verbose):
        super().cache_load_and_save(base_path, op, verbose)
        svd_encodings_path = base_path/'svd_encodings.pt'
        
        if op == 'load':
            self._svd_encodings = torch.load(str(svd_encodings_path))
        elif op == 'save':
            if verbose: print(f'{self.split} SVD encodings cache does not exist! Cacheing...', flush=True)
            self.calculate_all_svd_encodings(verbose=verbose)
            torch.save(self._svd_encodings, str(svd_encodings_path))
            if verbose: print(f'Saved {self.split} SVD encodings cache to disk.', flush=True)
        else:
            raise ValueError(f'Unknown operation: {op}')


@nb.njit
def calculate_svd_encodings(edges, num_nodes, calculated_dim):
    adj = np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for i in range(edges.shape[0]):
        adj[nb.int64(edges[i,0]),nb.int64(edges[i,1])] = 1
    
    for i in range(num_nodes):
        adj[i,i] = 1
    u, s, vh = np.linalg.svd(adj)
    
    if calculated_dim < num_nodes:
        s = s[:calculated_dim]
        u = u[:,:calculated_dim]
        vh = vh[:calculated_dim,:]
        
        encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)
    elif calculated_dim > num_nodes:
        z = np.zeros((num_nodes,calculated_dim-num_nodes,2),dtype=np.float32)
        encodings = np.concatenate((np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1), z), axis=1)
    else:
        encodings = np.stack((u,vh.T),axis=-1) * np.expand_dims(np.sqrt(s), axis=-1)
    return encodings


class SVDEncodingsGraphDataset(SVDEncodingsDatasetBase, GraphDataset):
    def calculate_encodings(self, item):
        num_nodes = int(item[self.num_nodes_key])
        edges = item[self.edges_key]
        encodings = calculate_svd_encodings(edges, num_nodes, self.calculated_dim)
        return encodings


