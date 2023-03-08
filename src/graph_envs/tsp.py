import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random


class TSPEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True, return_graph_obs=False, is_eval_env=False) -> None:
        super(TSPEnv, self).__init__()
        
        self.NODE_TAKEN = 0
        self.EDGE_WEIGHT = 0
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(n_nodes+2*n_edges+2*n_edges*2,))
        
        self.return_graph_obs = return_graph_obs

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
            # delay = np.random.randint(1, 3, size=(self.n_nodes, self.n_nodes))*1.0
        else:
            delay = np.random.randint(10, 11, size=(self.n_nodes, self.n_nodes))/10.0
            
        
        for u, v, d in G.edges(data=True):
            d['weight'] = delay[u, v]
        
        G = G.to_directed()
        
        
        x = np.zeros((self.n_nodes, 1), dtype=np.float32)
        
        self.head = 0
        x[self.head, self.NODE_TAKEN] = 1
        
        self.adj = nx.adjacency_matrix(G, weight='weight').todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['weight'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
         
        
        self.optimal_solution = 0
        if self.is_eval_env:
            self.optimal_cycle = nx.approximation.traveling_salesman_problem(G, weight='weight')
            for i in range(len(self.optimal_cycle)-1):
                self.optimal_solution += G[self.optimal_cycle[i]][self.optimal_cycle[i+1]]['weight']
        
        self.total_solution_cost = 0
        
        
        info = {'mask': self._get_mask()}
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        return self._vectorize_graph(self.graph), info
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((self.n_nodes,), dtype=bool)
        mask[self._get_neighbors(self.head)] = 1
        mask[self.graph.nodes[:, self.NODE_TAKEN] == 1] = 0
        return mask
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        assert action in self._get_neighbors(self.head), f"Node {action} is not a neighbor of the current path head {self.head}!"
        assert np.isclose(self.graph.nodes[action, self.NODE_TAKEN], 1) == False, f"Node {action} is already a part of the path!"
        
        
        done = False
        reward = 1-self.adj[self.head, action]
        self.total_solution_cost += self.adj[self.head, action]
        info = {}
        
        self.graph.nodes[action, self.NODE_TAKEN] = 1
        self.head = action
        
        
        if np.isclose(self.graph.nodes[:, self.NODE_TAKEN], 0).any() == False:    
            done = True
            reward += self.num_nodes
            info['solved'] = True
        
        
        info['mask'] = self._get_mask()
        if (not done) and (info['mask'].sum() == 0):
            done = True
            # reward -= np.isclose(self.graph.nodes[:, self.NODE_TAKEN], 0).sum()
            info['solved'] = False
            
        
        if done:
            info['heuristic_solution'] = self.optimal_solution
            info['solution_cost'] = self.total_solution_cost
                
        return self._vectorize_graph(self.graph), reward, done, False, info
        