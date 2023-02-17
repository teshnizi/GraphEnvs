import gymnasium as gym 
from src.graph_envs import shortest_path
import numpy as np


def test_shortest_path():
    for tst in range(1, 5):
        
        n_dests = np.random.randint(1, 5)
        
        env = gym.make('SteinerTree-v0', n_nodes=5, n_edges=10, n_dests=tst)

        for _ in range(5):
            print('---------')
            
            obs, info = env.reset()
            mask = info['mask']
            print(obs)
            
            done = False
            assert (obs.nodes == env.IS_TARGET).sum() == tst
            
            while not done:
                valid_actions = mask.nonzero()[0]
                action = np.random.choice(valid_actions)
                print(action)
                obs, reward, done, _, info = env.step(action)
                mask = info['mask']
            print(info['generated_solution'])
        assert True
        
        
if __name__ == '__main__':
    test_shortest_path()