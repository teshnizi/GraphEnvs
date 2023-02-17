import gymnasium as gym 
from src.graph_envs import shortest_path
import numpy as np


def test_shortest_path():
    env = gym.make('ShortestPath-v0', n_nodes=5, n_edges=7)

    for _ in range(5):
        
        obs, info = env.reset()
        mask = info['mask']
        done = False
        
        while not done:
            valid_actions = mask.nonzero()[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, _, info = env.step(action)
            mask = info['mask']
    assert True