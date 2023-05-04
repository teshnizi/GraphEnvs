# GraphEnvs
Graph Reinforcement Learning (RL) Environments

## Installation

First install [torch](https://pytorch.org/) and [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The simply use pip to install graph_envs:
```bash
  pip install graph-envs
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
| Largest Clique   | ✅ (Min Vertex Cover) | $v \in \mathcal{V}$ |
| Densest Subgraph   | ✅  | $v \in \mathcal{V}$ |
| Node Coloring  | 🛠️  | $(v, c) \in \mathcal{V} \times \mathbb{Z} $ |

### GraphEnvs-Extended:

| Environment      | Developed |  Action Space  |
| :----: |    :----:   | :-------:|
| MultiCast Routing   | ✅   | $e \in \mathcal{E}$|
| Distribution Center Selection  | ✅   | $v \in \mathcal{V}$ |
| Persihable Product Delivery   | ✅   | $v \in \mathcal{V}$|
| Public Transport Navigation  | 🛠️   | - |



  
## Example 

```python
import gymnasium as gym 
import graph_envs
import numpy as np

env = gym.make('LongestPath-v0',
               n_nodes=10,
               n_edges=20,
               weighted=True,
               is_eval_env=True, 
               parenting=2
               )

for sd in range(0, 10):

    print(f'===== {sd} =====')
    obs, info = env.reset(seed=sd)
    mask = info['mask']
    done = False
   
    while not done:
        valid_actions = mask.nonzero()[0]
        action = np.random.choice(valid_actions)        
        obs, reward, done, _, info = env.step(action)
        print('Valid actions:', valid_actions, '  Action:', action, '  Reward:', reward, '  Done:', done)
        mask = info['mask']
        
    print(info['solution_cost'], info['solved'], info['heuristic_solution'])

```

## Environment Details:

1. **Shortest Path**: The goal is to find the shortest path from the source node to the target node. At each step, an edge is added to the path. The episode is over when we reach the target node.
2. **Steiner Tree**: The goal is to find the tree with the minimum weight that connects a source node to a number of destination nodes.


