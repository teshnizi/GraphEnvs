import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import networkx as nx 
import random

from typing import Tuple

class SteinerTreeEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
        - n_dests: number of destinations to be reached
    '''
    
    def __init__(self, n_nodes, n_edges, n_dests=3, weighted=True) -> None:
        super(SteinerTreeEnv, self).__init__()
        
        # Node status codes
        self.HAS_NOTHING = np.array([1.0], dtype=np.float32)
        self.IS_TAKEN = np.array([2.0], dtype=np.float32)
        self.IS_TARGET = np.array([3.0], dtype=np.float32)
        
        # Edge status codes
        self.IS_NOT_TAKEN = np.array([1.0], dtype=np.float32)
        self.IS_TAKEN = np.array([2.0], dtype=np.float32)
        
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_dests = n_dests
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
        
       
        x = np.zeros((self.n_nodes, 1), dtype=np.float32) + self.HAS_NOTHING
        self.dests = np.random.choice(self.n_nodes, size=self.n_dests+1, replace=False)
        

        self.approx_solution = nx.algorithms.approximation.steinertree.steiner_tree(G, self.dests, weight='delay', method='kou')
        G = G.to_directed()
        
        self.src, self.dests = self.dests[0], self.dests[1:]
        
        
        x[self.src] = self.IS_TAKEN
        x[self.dests] = self.IS_TARGET
        
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        self.generated_solution = []
        
        info = {'mask': self._get_mask()}
        
        return self.graph, info
    
    # def _get_neighbors(self, node):
    #     neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
    #     return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((2 * self.n_edges,), dtype=bool) < 1.0
        mask[self.graph.nodes.flatten()[self.graph.edge_links[:, 0]] != self.IS_TAKEN] = False
        mask[self.graph.nodes.flatten()[self.graph.edge_links[:, 1]] == self.IS_TAKEN] = False
        return mask
    
    def step(self, action: Tuple[int, int]) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        u, v = self.graph.edge_links[action, 0], self.graph.edge_links[action, 1]
        
        assert (u < self.n_nodes), f"Node {u} is out of bounds!"
        assert (v < self.n_nodes), f"Node {v} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert np.isclose(self.graph.edges[action], self.IS_TAKEN) == False, f"BUG IN THE LIBRARY! Edge {action} is already a part of the path!"
        assert np.isclose(self.graph.nodes[u], self.IS_TAKEN) == True, f"BUG IN THE LIBRARY! Node {u} is not a part of the path!"
        assert np.isclose(self.graph.nodes[v], self.IS_TAKEN) == False, f"BUG IN THE LIBRARY! Node {v} is already a part of the path!"
        
        done = False
        reward = -self.graph.edges[action, 0]
        self.graph.nodes[v] = self.IS_TAKEN
        self.generated_solution.append((u, v))
        
        if np.sum(self.graph.nodes == self.IS_TARGET) == 0:
            done = True
            
        info = {'mask': self._get_mask()}
            
        if done:
            info['approx_solution'] = self.approx_solution
            info['generated_solution'] = self.generated_solution
        else:
            assert info['mask'].sum() > 0, "No more actions possible! Shouldn't happen!"
        
        return self.graph, reward, done, False, info
        