import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import networkx as nx 
import random

from typing import Tuple


def get_prime_numbers(n):
    '''
    Returns a list of prime numbers less than n
    '''
    prime_numbers = []
    for num in range(2, n):
        if all(num % i != 0 for i in range(2, num)):
            prime_numbers.append(num)
    return prime_numbers

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
        # self.HAS_NOTHING = np.array([0.0], dtype=np.float32)
        # self.HAS_MSG = np.array([2.0], dtype=np.float32)
        # self.IS_TARGET = np.array([3.0], dtype=np.float32)
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        
        # Edge status codes
        # self.IS_NOT_TAKEN = np.array([1.0], dtype=np.float32)
        # self.HAS_MSG = np.array([2.0], dtype=np.float32)
        self.EDGE_IS_TAKEN = 1
        self.EDGE_WEIGHT = 0
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_dests = n_dests
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_edges)
        # self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(n_nodes+2*n_edges+2*n_edges*2,))
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(2*n_nodes+2*n_edges*2+2*n_edges*2,))
        # self.observation_space = gym.spaces.Graph(
        #     node_space=gym.spaces.Box(low=0, high=4, shape=(1,)), 
        #     edge_space=gym.spaces.Box(low=0, high=1, shape=(1,)),
        #     )
        
        

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
        
       
        # x = np.zeros((self.n_nodes, 1), dtype=np.float32) + self.HAS_NOTHING
        x = np.zeros((self.n_nodes, 2), dtype=np.float32)      
        
        self.dests = np.random.choice(self.n_nodes, self.n_dests+1, replace=False)
        
        if self.n_dests == 1:
            self.approx_solution = nx.algorithms.shortest_paths.generic.shortest_path_length(G, self.dests[0], self.dests[1], weight='delay')
        else:
            approx_solution_graph = nx.algorithms.approximation.steinertree.steiner_tree(G, self.dests, weight='delay', method='kou')  
            self.approx_solution = sum([G[u][v]['delay'] for u, v in approx_solution_graph.edges()])
            
        G = G.to_directed()
        
        self.src, self.dests = self.dests[0], self.dests[1:]
        
        
        x[self.src, self.NODE_HAS_MSG] = 1
        x[self.dests, self.NODE_IS_TARGET] = 1
        
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
       
        edge_f = np.concatenate((edge_f, np.zeros((2*self.n_edges, 1), dtype=np.float32)), axis=1)
        
      
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        self.generated_solution = []

        info = {'mask': self._get_mask()}
        self._vectorize_graph(self.graph)
        return self._vectorize_graph(self.graph), info
    
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_mask(self) -> np.array:
        mask = np.zeros((2 * self.n_edges,), dtype=bool) < 1.0
        mask[self.graph.nodes[self.graph.edge_links[:, 0], self.NODE_HAS_MSG] < 0.5] = False
        mask[self.graph.nodes[self.graph.edge_links[:, 1], self.NODE_HAS_MSG] > 0.5] = False
        return mask
    
    
    def step(self, action: Tuple[int, int]) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        
        u, v = self.graph.edge_links[action, 0], self.graph.edge_links[action, 1]
        
        # print(self.graph.nodes.T)
        assert (u < self.n_nodes), f"Node {u} is out of bounds!"
        assert (v < self.n_nodes), f"Node {v} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert np.isclose(self.graph.edges[action, self.EDGE_IS_TAKEN], 1) == False, f"BUG IN THE LIBRARY! Edge {action} is already a part of the path!"
        assert np.isclose(self.graph.nodes[u, self.NODE_HAS_MSG], 1), f"BUG IN THE LIBRARY! Node {u} is not a part of the path!"
        assert np.isclose(self.graph.nodes[v, self.NODE_HAS_MSG], 0), f"BUG IN THE LIBRARY! Node {v} is already a part of the path!"
        
        done = False
        reward = -self.graph.edges[action, self.EDGE_WEIGHT]
        self.graph.nodes[v, self.NODE_HAS_MSG] = 1
        self.generated_solution.append((u, v))
        
        
        # print(self.graph.nodes[:, self.NODE_IS_TARGET] == 1)
        # print(self.graph.nodes[:, self.NODE_HAS_MSG] == 0)
        # print(np.logical_and(self.graph.nodes[:, self.NODE_HAS_MSG] == 0, self.graph.nodes[:, self.NODE_IS_TARGET] == 1))
         
        if np.logical_and(self.graph.nodes[:, self.NODE_HAS_MSG] == 0, self.graph.nodes[:, self.NODE_IS_TARGET] == 1).sum() == 0:
            done = True
   
            
        info = {'mask': self._get_mask()}
        
        if done:
            info['heuristic_solution'] = self.approx_solution
            info['solved'] = True
            pass
        else:
            assert info['mask'].sum() > 0, "No more actions possible! Shouldn't happen!"
        
        
        return self._vectorize_graph(self.graph), reward, done, False, info
        