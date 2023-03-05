import gymnasium as gym 
import numpy as np
import warnings
import torch
import sys 
import os 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.graph_envs import shortest_path, utils

env_id = 'MaxIndependentSet-v0'
kwargs = {'n_nodes': 5, 'n_edges': 7, 'return_graph_obs': True, 'weighted': False}

def test_max_independent_set():
    
    env = gym.make(env_id, **kwargs)
    
    for _ in range(5):
        
        obs, info = env.reset()
        
        mask = info['mask']
        done = False
        graph_obs = info['graph_obs']

        obs = torch.Tensor(obs.reshape(1, -1))
        nodes, edges, edge_links = utils.devectorize_graph(vector=obs, env_id=env_id, **kwargs)
        
        assert (nodes == torch.Tensor(graph_obs.nodes)).all(), f"{obs.nodes} {graph_obs.nodes}"
        assert (edges == torch.Tensor(graph_obs.edges)).all(), f"{obs.edges} {graph_obs.edges}"
        assert (edge_links == torch.Tensor(graph_obs.edge_links)).all(), f"{obs.edge_links} {graph_obs.edge_links}"
        assert (nodes[:, :, 1]).abs().sum() == 0, nodes[:, :, 1]
        
        
        while not done:
            valid_actions = mask.nonzero()[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, _, info = env.step(action)
            nodes, edges, edge_links = utils.devectorize_graph(vector=torch.Tensor(obs.reshape(1, -1)), env_id=env_id, **kwargs)
            mask = info['mask']
            
    assert True
    
if __name__ == '__main__':
    test_max_independent_set()