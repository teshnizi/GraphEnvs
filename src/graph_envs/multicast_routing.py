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
    Environment for multicast routing problem.
    '''
    
    def __init__(self, n_nodes, n_edges=-1, n_dests=3, weighted=True, max_distance=-1, is_eval_env=False) -> None:
        super(SteinerTreeEnv, self).__init__()
        
        # Node status codes
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        self.NODE_MAX_DISTANCE = 2
        
        # Edge status codes
        self.EDGE_WEIGHT = 0
        self.EDGE_IS_TAKEN = 1
        
        self.n_nodes = n_nodes
        if n_edges == -1:
            n_edges = (n_nodes * (n_nodes - 1) // 2) * 0.30

        self.n_edges = n_edges
        self.n_dests = n_dests
        self.weighted = weighted
        
        #TODO: This should be a parameter
        if max_distance == -1:
            max_distance = np.log(n_nodes) * (1+0.3)/2
            
        self.max_distance = max_distance
            
        self.action_space = gym.spaces.Discrete(n_edges)
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(2*n_nodes+2*n_edges*2+2*n_edges*2,))
    
        self.is_eval_env = is_eval_env
        
    def reset(self, seed=None, options={}) -> np.array:
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        
        while True:
            G = nx.gnm_random_graph(self.n_nodes, self.n_edges)
            if nx.is_connected(G):
                break
        
        if self.weighted:
            delay = np.random.randint(3, 10, size=(self.n_nodes, self.n_nodes))/10.0
        else:
            delay = np.random.randint(1, 2, size=(self.n_nodes, self.n_nodes))/1.0
        
        for u, v, d in G.edges(data=True):
            d['delay'] = delay[u, v]
        
       
        # x = np.zeros((self.n_nodes, 1), dtype=np.float32) + self.HAS_NOTHING
        x = np.zeros((self.n_nodes, 2), dtype=np.float32)      
        
        self.src = 0
        self.dests = np.random.choice(np.arange(1, self.n_nodes), size=self.n_dests, replace=False)
        # self.src, self.dests = self.dests[0], self.dests[1:]
        
        self.approx_solution = 0
        
        if self.is_eval_env:
            # if self.n_dests == 1:
            #     self.approx_solution = nx.algorithms.shortest_paths.generic.shortest_path_length(G, self.dests[0], self.dests[1], weight='delay')
            # elif self.n_dests == self.n_nodes - 1:
            #     self.approx_solution = sum([features['delay'] for _, _, features in nx.minimum_spanning_edges(G, weight='delay')])
            # else:
            #     approx_solution_graph = nx.algorithms.approximation.steinertree.steiner_tree(G, self.dests, weight='delay', method='kou')  
            #     self.approx_solution = sum([G[u][v]['delay'] for u, v in approx_solution_graph.edges()])
            self.approx_solution = -1
            
        G = G.to_directed()
        self.G = G
        
        
        x[self.src, self.NODE_HAS_MSG] = 1
        x[self.dests, self.NODE_IS_TARGET] = 1
        
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
       
        edge_f = np.concatenate((edge_f, np.zeros((2*self.n_edges, 1), dtype=np.float32)), axis=1)
        
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        self.generated_solution = []

        self.solution_cost = 0
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
        self.solution_cost -= reward
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
            info['solution_cost'] = self.solution_cost
            pass
        else:
            assert info['mask'].sum() > 0, "No more actions possible! Shouldn't happen!"
        
        
        return self._vectorize_graph(self.graph), reward, done, False, info
        