import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random

from graph_envs.utils import vectorize_graph
import graph_envs.feature_extraction as fe

class ShortestPathEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True, return_graph_obs=False, parenting=-1, structural_features=True, is_eval_env=False) -> None:
        super(ShortestPathEnv, self).__init__()
        
        assert parenting == -1, "Parenting is not available for shortest path"
        
        # self.HAS_NOTHING = np.array([0.0], dtype=np.float32)
        
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        
        # self.IS_TAKEN = np.array([1.0], dtype=np.float32)
        # self.IS_TARGET = np.array([1.0], dtype=np.float32)
        
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)

        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=((2+fe.get_num_features())*n_nodes+2*n_edges+2*n_edges*2,))
        self.return_graph_obs = return_graph_obs
        self.solution_cost = 0
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
            # delay = np.random.randint(1, 3, size=(self.n_nodes, self.n_nodes))*1.0
        else:
            delay = np.random.randint(10, 11, size=(self.n_nodes, self.n_nodes))/10.0
            
        
        for u, v, d in G.edges(data=True):
            d['delay'] = delay[u, v]
        
        
        G = G.to_directed()
        
        x = np.zeros((self.n_nodes, 2 + fe.get_num_features()), dtype=np.float32)
        
        self.src, self.dest = np.random.choice(self.n_nodes, size=2, replace=False)
        x[self.src, self.NODE_HAS_MSG], x[self.dest, self.NODE_IS_TARGET] = 1, 1
        
        # Adding structural features
        sf = fe.generate_features(G)
        x[:, -fe.get_num_features():] = sf
        
        self.head = self.src
        self.adj = nx.adjacency_matrix(G, weight='delay').todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        self.optimal_solution = 0
        if self.is_eval_env:
            self.optimal_solution = nx.shortest_path_length(G, source=self.src, target=self.dest, weight='delay')
        
        info = {'mask': self._get_mask()}
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        self.solution_cost = 0
        return vectorize_graph(self.graph), info
    
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((self.n_nodes,), dtype=bool)
        mask[self._get_neighbors(self.head)] = 1
        mask[self.graph.nodes[:, self.NODE_HAS_MSG] == 1] = 0
        return mask
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        assert action in self._get_neighbors(self.head), f"Node {action} is not a neighbor of the current path head {self.head}!"
        assert np.isclose(self.graph.nodes[action, self.NODE_HAS_MSG], 1) == False, f"Node {action} is already a part of the path!"
        
        done = False
        reward = -self.adj[self.head, action]
        self.solution_cost -= reward
        info = {}
        
        if np.isclose(self.graph.nodes[action, self.NODE_IS_TARGET], 1):
            # reward += self.n_nodes
            done = True
            info['solved'] = True
        
        
        self.graph.nodes[action, self.NODE_HAS_MSG] = 1
        self.head = action
        
        info['mask'] = self._get_mask()
        if (not done) and (info['mask'].sum() == 0):
            done = True
            reward = -self.n_nodes
            info['solved'] = False
            
        if done:
            info['heuristic_solution'] = self.optimal_solution
            info['solution_cost'] = self.solution_cost
        return vectorize_graph(self.graph), reward, done, False, info
        