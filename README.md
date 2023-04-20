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
from src.graph_envs import shortest_path
import numpy as np

env = gym.make('ShortestPath-v0', n_nodes=5, n_edges=7)

for _ in range(5):
    
    obs, info = env.reset()
    mask = info['mask']
    done = False
    
    while not done:
        valid_actions = mask.nonzero()[0]
        action = np.random.choice(valid_actions)
        obs, reward, done, _, info = env.step(action)
        print(obs, rewards, done)
        mask = info['mask']

```


## Supported Environments

### GraphEnvs-Basic:

| Environment      | Developed |  Action Space  |
| :----: |    :----:   | :-------:|
| Shortest Path      | ✅       | $v \in \mathcal{V}$   |
| Steiner Tree   | ✅   | $e \in \mathcal{E}$      |
| MST   | ✅  | $e \in \mathcal{E}$      |
| Minimum Vertex Cover  | ✅   | $v \in \mathcal{V}$ |
| TSP   | ✅   | $v \in \mathcal{V}$ |
| Longest Path   | ✅   | $v \in \mathcal{V}$ |
| Largest Clique   | :heavy_multiplication_x:   | $v \in \mathcal{V}$ |
| Densest Subgraph   | :heavy_multiplication_x:  | $v \in \mathcal{V}$ |
| Node Coloring  | :heavy_multiplication_x:  | $(v, c) \in \mathcal{V} \times \mathbb{Z} $ |

### GraphEnvs-Extended:

| Environment      | Developed |  Action Space  |
| :----: |    :----:   | :-------:|
| Public Transport Navigation  | :heavy_multiplication_x:   | - |
| MultiCast Routing   | ✅   | $e \in \mathcal{E}|
| Distribution Center Selection  | ✅   | $v \in \mathcal{V}$ |


1. **Shortest Path**: The goal is to find the shortest path from the source node to the target node. At each step, an edge is added to the path. The episode is over when we reach the target node.
2. **Steiner Tree**: The goal is to find the tree with the minimum weight that connects a source node to a number of destination nodes.


