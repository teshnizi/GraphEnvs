
import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np 
import random 
import networkx as nx

from typing import Tuple, SupportsFloat

class DistributionCenterEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the nodes are weighted or not
    '''
    
    def find_nodes_in_range(self, node, distance, G):
        return np.array(list(nx.single_source_dijkstra_path_length(G, node, cutoff=distance, weight='delay').keys()))
        
        
    def __init__(self, n_nodes, n_edges, weighted=True, max_distance=1, target_count=-1, return_graph_obs=False, is_eval_env=False, parenting='Advanced') -> None:
        super(DistributionCenterEnv, self).__init__()
        
        assert parenting in ['Basic', 'Advanced']
        
        self.NODE_WEIGHT = 0
        self.NODE_IS_TAKEN = 1
        self.NODE_IS_TARGET = 2
        self.NODE_IS_COVERED = 3
        self.NODE_MAX_DISTANCE = 4
        
        self.EDGE_WEIGHT = 0
        
        if target_count == -1:
            self.target_count = n_nodes//5
        else:
            self.target_count = target_count
            
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        self.parenting = parenting
        self.max_distance = max_distance
        
        self.action_space = gym.spaces.Discrete(n_nodes)
        
        self.return_graph_obs = return_graph_obs
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(5*n_nodes+2*n_edges+2*n_edges*2,))
        self.is_eval_env = is_eval_env
        
        
    def reset(self, seed=None, options={}) -> np.array:
        
        if seed != None:
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
        
        cost = np.random.randint(1, 4, size=(self.n_nodes))/1.0
        
        for u in G.nodes():
            G.nodes[u]['cost'] = cost[u]
        
        targets = np.random.choice(self.n_nodes, size=self.target_count, replace=False)
        
        
        # TODO: add a baseline  
        self.approx_solution = -1
        
        G = G.to_directed()
        self.nx_graph = G
        
        
        x = np.zeros((self.n_nodes, 5), dtype=np.float32)
        x[:, self.NODE_WEIGHT] = cost
        x[targets, self.NODE_IS_TARGET] = 1
        x[:, self.NODE_MAX_DISTANCE] = self.max_distance
        
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        self.in_range_dict = {}
        
        for t in targets:
            self.in_range_dict[t] = self.find_nodes_in_range(t, self.max_distance, G)
        
        
        info = {'mask': self._get_mask()}
        
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        self.solution_cost = 0
        return self._vectorize_graph(self.graph), info
    
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_mask(self) -> np.array:
        
        mask = np.zeros((self.n_nodes,), dtype=bool)
        targets_left = np.logical_and(self.graph.nodes[:, self.NODE_IS_TARGET] == 1, self.graph.nodes[:, self.NODE_IS_COVERED] == 0).nonzero()[0]
        if self.parenting == 'Advanced':
            for t in targets_left:
                mask[self.in_range_dict[t]] = True
        else:
            mask[:] = True
            
        mask[self.graph.nodes[:, self.NODE_IS_TAKEN] == 1] = False
        
        return mask
    

    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert np.isclose(self.graph.nodes[action, self.NODE_IS_TAKEN], 0), f"Node {action} is already taken!"
        
        done = False
        reward = -self.graph.nodes[action, self.NODE_WEIGHT]
        self.solution_cost -= reward
        info = {}
        
        self.graph.nodes[action, self.NODE_IS_TAKEN] = 1
        new_covered = self.find_nodes_in_range(action, self.max_distance, self.nx_graph)
        for nd in new_covered:
            if self.graph.nodes[nd, self.NODE_IS_COVERED] == 1:
                continue
            self.graph.nodes[nd, self.NODE_IS_COVERED] = 1
            if self.graph.nodes[nd, self.NODE_IS_TARGET] == 1:
                reward += 1
                
        
        info['mask'] = self._get_mask()
        
        targets_left = np.logical_and(self.graph.nodes[:, self.NODE_IS_TARGET] == 1, self.graph.nodes[:, self.NODE_IS_COVERED] == 0).nonzero()[0]
        if len(targets_left) == 0:
            done = True
            info['solved'] = True      
        
        if done:
            info['heuristic_solution'] = self.approx_solution
            info['solution_cost'] = self.solution_cost
        return self._vectorize_graph(self.graph), reward, done, False, info
          
        
        
        
            