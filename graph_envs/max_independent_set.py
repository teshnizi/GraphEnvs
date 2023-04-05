
import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np 
import random 
import networkx as nx

from typing import Tuple, SupportsFloat

class MaxIndependentSet(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the nodes are weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True, return_graph_obs=False, is_eval_env=False) -> None:
        super(MaxIndependentSet, self).__init__()
        
        
        self.NODE_WEIGHT = 0
        self.NODE_IS_TAKEN = 1
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)
        
        self.return_graph_obs = return_graph_obs
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(2*n_nodes+2*n_edges+2*n_edges*2,))
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
            cost = np.random.randint(3, 10, size=(self.n_nodes))/10.0
            
        else:
            cost = np.random.randint(1, 2, size=(self.n_nodes))/1.0
            
        for u in G.nodes():
            G.nodes[u]['cost'] = cost[u]
            
        self.approx_solution = 0
        if self.is_eval_env:
            if not self.weighted:
                self.approx_solution = len(nx.approximation.maximum_independent_set(G))
            else:
                self.approx_solution = -1
            
        G = G.to_directed()
        
        x = np.zeros((self.n_nodes, 2), dtype=np.float32)
        x[:, self.NODE_WEIGHT] = cost
        
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([1.0 for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        info = {'mask': self._get_mask()}
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        self.solution_cost = 0
        return self._vectorize_graph(self.graph), info
    
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_mask(self) -> np.array:
        
        # self.graph.nodes[3,1] = 1
        # self.graph.nodes[1,1] = 1
        
        mask = np.zeros((self.n_nodes,), dtype=bool)
        mask[self.graph.nodes[:, self.NODE_IS_TAKEN] == 0] = True
        
        return mask

    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        assert np.isclose(self.graph.nodes[action, self.NODE_IS_TAKEN], 0), f"Node {action} is already a part of the path!"
        
        done = False
        reward = -self.graph.nodes[action, self.NODE_WEIGHT]
        self.solution_cost -= reward
        info = {}
        
        self.graph.nodes[action, self.NODE_IS_TAKEN] = 1
        info['mask'] = self._get_mask()
        
        if info['mask'].sum() == 0:
            done = True
            info['solved'] = True      
        
        if done:
            info['heuristic_solution'] = self.approx_solution
            info['solution_cost'] = self.solution_cost
        
        return self._vectorize_graph(self.graph), reward, done, False, info
          
        
        
        
            