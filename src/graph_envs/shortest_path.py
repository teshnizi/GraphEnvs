import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import networkx as nx 
import random

from typing import Tuple

class ShortestPathEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True) -> None:
        super(ShortestPathEnv, self).__init__()
        
        self.HAS_NOTHING = np.array([1.0], dtype=np.float32)
        self.IS_TAKEN = np.array([2.0], dtype=np.float32)
        self.IS_TARGET = np.array([3.0], dtype=np.float32)
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)
        # self.observation_space = gym.spaces.Box(low=-1., high=11., shape=(2, n_nodes, n_nodes))        
        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(low=0, high=4, shape=(1,)), 
            edge_space=gym.spaces.Box(low=0, high=1, shape=(1,)),
            )

    def reset(self, seed=None, options={}) -> np.array:
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        
        while True:
            G = nx.gnm_random_graph(self.n_nodes, self.n_edges)
            if nx.is_connected(G):
                break
        
        if self.weighted:
            delay = np.random.randint(1, 10, size=(self.n_nodes, self.n_nodes))/10.0
        else:
            delay = np.random.randint(10, 11, size=(self.n_nodes, self.n_nodes))/10.0
        
        for u, v, d in G.edges(data=True):
            d['delay'] = delay[u, v]
        
        G = G.to_directed()
        
        x = np.zeros((self.n_nodes, 1), dtype=np.float32) + self.HAS_NOTHING
        self.src, self.dest = np.random.choice(self.n_nodes, size=2, replace=False)
        x[self.src], x[self.dest] = self.IS_TAKEN, self.IS_TARGET
        self.head = self.src
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
         
        self.optimal_solution = nx.shortest_path_length(G, source=self.src, target=self.dest, weight='delay')
        
        info = {'mask': self._get_mask()}
        
        return self.graph, info
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((self.n_nodes,), dtype=bool)
        mask[self._get_neighbors(self.head)] = 1
        mask[self.graph.nodes[:, 0] == self.IS_TAKEN] = 0
        return mask
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        assert action in self._get_neighbors(self.head), f"Node {action} is not a neighbor of the current path head {self.head}!"
        assert np.isclose(self.graph.nodes[action], self.IS_TAKEN) == False, f"Node {action} is already a part of the path!"
        
        
        done = False
        reward = -self.adj[self.head, action]
        
        if np.isclose(self.graph.nodes[action], self.IS_TARGET):
            done = True
            
        self.graph.nodes[action] = self.IS_TAKEN
        self.head = action
        
        info = {'mask': self._get_mask()}
        if info['mask'].sum() == 0:
            done = True
            reward = -10
            
        if done:
            info['optimal_solution'] = self.optimal_solution
        
        return self.graph, reward, done, False, info
        