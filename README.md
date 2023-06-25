# GraphEnvs
Graph Reinforcement Learning (RL) Environments

## Installation

First install [torch](https://pytorch.org/) and [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Then simply use pip to install graph_envs:
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

- **Shortest Path**: The goal of the agent in this environment is to find the shortest path from a given source to a given destination. The agent does so by taking a node at each step. The taken node is added to the end of an existing path, until the path reaches the destination or is not expandable anymore. The initial partial path is of length 0, and only consists of the source node. The reward in each step is equal to negative of the weight of the edge added to the path.
    
- **Steiner Tree**: In this environment, a single source and a set of destinations are given, and the agent's goal is to find the a sub-tree with minimum total edge weight in the graph that connects the source to all of the destinations. At each step, the agent picks an edge, and that edge is added to the existing partial tree. The episode continues until the partial tree reaches all of the destinations. The initial partial tree has 1 node (the source node) and 0 edges. The reward in each step is equal to negative of the weight of the taken edge.
    
- **MST**: In this environment, the agent's goal is to find a tree with minimum weight that connects all of the nodes to each other. At each step, the agent picks an edge, and that edge is added to the existing partial tree. The initial tree has 1 node (node with id $0$) and 0 edges. To use this environment, simply set n_dests variable of the Steiner tree environment equal to $N-1$, where $N$ is the number of nodes in the graph. The reward in each step is equal to negative of the weight of the taken edge.

- **TSP**: In the TSP environment, the agent should find a cycle with minimum length that connects all of the cities, and passes each city exactly once. Only edges given in the input are available to the agent, and the problem might not have a solution. The initial solution is a path of length 0 only consisting of node $0$. In each step, the agent picks a node and adds it to the path, and eventually it can close the path if all of the nodes in the graph are taken. The reward in each step is equal to negative of the weight of the taken edge. The agent is penalized with a large negative reward if the final solution is infeasible.

- **Hamiltonian Cycle**: In this environment, the goal is to find a cycle that connects all of the nodes to each other, and passes through each node exactly once. The environment dynamics are similar to the TSP environment. To use this environment, use the TSP environment with weighted=False.

- **Minimum Vertex Cover**: In this environment, a set of nodes with different costs are given. The goal is to select a set of nodes with minimal total cost such that every other node in the graph is connected to at least one of the selected nodes. The agent takes a node at each step to add to the solution. The episode stops when all nodes are covered. The reward in each step is equal to negative of the cost of the taken node.

- **LongestPath**: The goal in this environment is to connect a given source to a given destination with the longest path possible. The path can not pass each node more than once. Dynamics are similar to the shortest path environment. The reward in each step is equal to of the weight of the edge added to the path.

- **Densest Subgraph**: The goal in this environment is to find a subset of nodes for which the induced subgraph has the highest ratio of edge count to node count. The partial solution is initially an empty graph. The agent takes a single node in each step, and adds it to the partial solution. Reward in each step is equal to change in the ratio.
    
- **Multicast Routing**: The goal in this environment is to connect a source node $s$ to a set of destinations $\{d_1, d_2, \dots, d_k\}$ with a tree with minimum total weight, while making sure all destinations are reachable from the source with a path not longer than $t$. This environment is similar to the Steiner tree environment, with addition of maximum delay constraints. Dynamics are also similar to the Steiner tree environment. By default, the reward for that step is equal to negative of the weight of the edge taken. For actions that connect the partial tree to a destination, a large negative (positive) number is added to the reward if the path between the source and the destination is more than (less than) $t$.

- **Distribution Center Selection**: In this environment, the agent is given a graph in which each node has a cost. There is a set of destination nodes $D$ and a max distance $t$. The agent has to choose a set $S$ of the nodes with minimum total cost such that every node in $D$ is accessible from at least one node in $S$ with a path not longer that $t$. Starting from an empty initial solution, the agent chooses $S$ by picking one node at each step and adding it to $S$. The reward in each step is equal to negative of cost of the taken node.

- **Perishable Product Delivery**: In this environment, the agent is given a set of pickup nodes, a dropoff node for each one of them, and a distance $t$. The goal is to find a trajectory for the agent in which it picks up the products from the pickup nodes and drops them off at their respective dropoff nodes without holding each product for more than $t$ time units. In each step, the agent can either take a node to go to from its current location, or pickup a product from its current node if it is a pickup node. The products are automatically delivered each time the agent reaches a destination and has the corresponding product.
