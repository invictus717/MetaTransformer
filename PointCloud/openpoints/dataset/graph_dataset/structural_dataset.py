import numpy as np
import numba as nb

from .graph_dataset import GraphDataset

NODE_FEATURES_OFFSET = 128
EDGE_FEATURES_OFFSET = 8

@nb.njit
def floyd_warshall(A):
    n = A.shape[0]
    D = np.zeros((n,n), dtype=np.int16)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                pass
            elif A[i,j] == 0:
                D[i,j] = 510
            else:
                D[i,j] = 1
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                old_dist = D[i,j]
                new_dist = D[i,k] + D[k,j]
                if new_dist < old_dist:
                    D[i,j] = new_dist
    return D

@nb.njit
def preprocess_data(num_nodes, edges, node_feats, edge_feats):
    node_feats = node_feats + np.arange(1,node_feats.shape[-1]*NODE_FEATURES_OFFSET+1,
                                            NODE_FEATURES_OFFSET,dtype=np.int16)
    edge_feats = edge_feats + np.arange(1,edge_feats.shape[-1]*EDGE_FEATURES_OFFSET+1,
                                            EDGE_FEATURES_OFFSET,dtype=np.int16)
    
    A = np.zeros((num_nodes,num_nodes),dtype=np.int16)
    E = np.zeros((num_nodes,num_nodes,edge_feats.shape[-1]),dtype=np.int16)
    for k in range(edges.shape[0]):
        i,j = edges[k,0], edges[k,1]
        A[i,j] = 1
        E[i,j] = edge_feats[k]
    
    D = floyd_warshall(A)
    return node_feats, D, E


class StructuralDataset(GraphDataset):
    def __init__(self,
                 distance_matrix_key      = 'distance_matrix',
                 feature_matrix_key       = 'feature_matrix',
                 **kwargs):
        super().__init__(**kwargs)
        self.distance_matrix_key      = distance_matrix_key      
        self.feature_matrix_key       = feature_matrix_key  
        
    def __getitem__(self, index):
        item = super().__getitem__(index)
        
        num_nodes = int(item[self.num_nodes_key])
        edges = item.pop(self.edges_key)
        node_feats = item.pop(self.node_features_key)
        edge_feats = item.pop(self.edge_features_key)
        
        node_feats, dist_mat, edge_feats_mat = preprocess_data(num_nodes, edges, node_feats, edge_feats)
        item[self.node_features_key] = node_feats
        item[self.distance_matrix_key] = dist_mat
        item[self.feature_matrix_key] = edge_feats_mat
        
        return item
        
