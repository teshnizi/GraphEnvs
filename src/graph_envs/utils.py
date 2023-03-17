import numpy as np 
import gymnasium as gym 
import torch 



def devectorize_graph(vector, env_id, **kwargs):
    bs = vector.shape[0]
    node_f, edge_f, _ = get_env_info(env_id)
    p1 = kwargs['n_nodes'] * node_f
    p2 = p1 + 2*kwargs['n_edges'] * edge_f
    
    x = vector[:, :p1].reshape(bs, kwargs['n_nodes'], node_f)
    edge_features = vector[:, p1:p2].reshape(bs, 2*kwargs['n_edges'], edge_f)
    edge_index = vector[:, p2:].reshape(bs, 2*kwargs['n_edges'], 2).long()
    
    return x, edge_features, edge_index
    

def to_pyg_graph(x, edge_features, edge_index):
    import torch_geometric as pyg
    graphs = [pyg.data.Data(x=x[i,:,:], edge_attr=edge_features[i,:,:], edge_index=edge_index[i,:,:].T) for i in range(x.shape[0])]
    batch = pyg.data.Batch.from_data_list(graphs).to(x.device)
    return batch


def get_env_info(env_id):
    if env_id == "ShortestPath-v0":
        node_f = 2
        edge_f = 1
        action_type = "node"
    elif env_id == "SteinerTree-v0":
        node_f = 2
        edge_f = 2
        action_type = "edge"
    elif env_id == "MaxIndependentSet-v0":
        node_f = 2
        edge_f = 1
        action_type = "node"
    elif env_id == "TSP-v0":
        node_f = 2
        edge_f = 1
        action_type = "node"
        
    return node_f, edge_f, action_type