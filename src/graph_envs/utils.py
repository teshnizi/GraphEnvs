import numpy as np 
import gymnasium as gym 
import torch 
import torch_geometric as pyg


def devectorize_graph(vector, env_id, **kwargs):
    bs = vector.shape[0]
    
    if env_id == "ShortestPath-v0":
        x = vector[:, :kwargs['n_nodes']].reshape(bs, kwargs['n_nodes'], 1)
        edge_features = vector[:, kwargs['n_nodes']:kwargs['n_nodes'] + 2*kwargs['n_edges']].reshape(bs, 2*kwargs['n_edges'], 1)
        edge_index = vector[:, kwargs['n_nodes'] + 2*kwargs['n_edges']:].reshape(bs, 2*kwargs['n_edges'], 2).long()
        
        return x, edge_features, edge_index


def to_pyg_graph(x, edge_features, edge_index):
    graphs = [pyg.data.Data(x=x[i,:,:], edge_attr=edge_features[i,:,:], edge_index=edge_index[i,:,:].T) for i in range(x.shape[0])]
    batch = pyg.data.Batch.from_data_list(graphs).to(x.device)
    return batch

