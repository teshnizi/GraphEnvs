import gymnasium as gym 
from src.graph_envs import shortest_path, utils
import numpy as np
import warnings

env_id = 'ShortestPath-v0'
kwargs = {'n_nodes': 5, 'n_edges': 7, 'return_graph_obs': True}

def test_shortest_path():
    
    env = gym.make(env_id, **kwargs)

    for _ in range(5):
        
        obs, info = env.reset()
        mask = info['mask']
        done = False
        graph_obs = info['graph_obs']
    
        obs = utils.devectorize_graph(vector=obs, env_id=env_id, **kwargs)
        
        assert (obs.nodes == graph_obs.nodes).all(), f"{obs.nodes} {graph_obs.nodes}"
        assert (obs.edges == graph_obs.edges).all(), f"{obs.edges} {graph_obs.edges}"
        assert (obs.edge_links == graph_obs.edge_links).all(), f"{obs.edge_links} {graph_obs.edge_links}"
        assert (obs.nodes == env.IS_TARGET).sum() == 1
        
        while not done:
            valid_actions = mask.nonzero()[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, _, info = env.step(action)
            obs = utils.devectorize_graph(vector=obs, env_id=env_id, **kwargs)
            assert (obs.nodes == graph_obs.nodes).all(), f"{obs.nodes} {graph_obs.nodes}"
            assert (obs.edges == graph_obs.edges).all(), f"{obs.edges} {graph_obs.edges}"
            assert (obs.edge_links == graph_obs.edge_links).all(), f"{obs.edge_links} {graph_obs.edge_links}"
        
            mask = info['mask']
    assert True