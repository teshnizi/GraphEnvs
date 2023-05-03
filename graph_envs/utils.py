import numpy as np 
import gymnasium as gym 
import torch 

import networkx as nx 

import matplotlib.pyplot as plt
from graph_envs.feature_extraction import get_num_features
import importlib

# import torch_geometric as pyg


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
    elif env_id == 'DistributionCenter-v0':
        node_f = 5
        edge_f = 1
        action_type = "node"
    elif env_id == 'MulticastRouting-v0':
        node_f = 4
        edge_f = 2
        action_type = "edge"
    elif env_id == 'LongestPath-v0':
        node_f = 2
        edge_f = 1
        action_type = "node"
    elif env_id == 'DensestSubgraph-v0':
        node_f = 1
        edge_f = 1
        action_type = "node"
    elif env_id == 'PerishableProductDelivery-v0':
        node_f = 1+3*5
        edge_f = 1
        action_type = "node"
    else:
        assert False, "Unknown env_id"
    
    node_f += get_num_features()
    return node_f, edge_f, action_type


def show_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    # draw edge labels: 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'delay'))
    
    # save to file:
    # plt.savefig("graph.png")
    plt.show()
    
    
def vectorize_graph(graph):
    return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
 