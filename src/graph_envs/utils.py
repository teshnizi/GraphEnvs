import numpy as np 
import gymnasium as gym

def devectorize_graph(vector, env_id, **kwargs):
    if env_id == "ShortestPath-v0":
        return gym.spaces.GraphInstance(
            vector[:kwargs['n_nodes']].reshape(-1, 1),
            vector[kwargs['n_nodes']:kwargs['n_nodes'] + 2*kwargs['n_edges']].reshape(-1, 1),
            vector[kwargs['n_nodes'] + 2*kwargs['n_edges']:].reshape(-1, 2).astype(int)
        )