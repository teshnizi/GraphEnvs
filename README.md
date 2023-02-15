# GraphEnvs
Graph Reinforcement Learning (RL) Environments

## Installation

First install [torch](https://pytorch.org/) and [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The simply use pip to install graph_envs:
```bash
  pip install graph-envs
```
  
## Example 

```python
import gymnasium as gym 
import graph_envs
import numpy as np

env = gym.make('ShortestPath-v0', n_nodes=10, n_edges=20)

for _ in range(5):
    obs, info = env.reset()
    mask = info['mask']
    
    done = False
    while not done:
        valid_actions = mask.nonzero()
        action = np.random.choice(len(valid_actions[0]))
        action = (valid_actions[0][action], valid_actions[1][action])
        
        obs, reward, done, _, info = env.step(action)
        print(obs, reward, done)
        mask = info['mask']

```


## Supported Environments

1. **Shortest Path**: The goal is to find the shortest path from the source node to the target node. At each step, an edge is added to the path. The episode is over when we reach the target node.
