import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random


class LongestPathEnv(gym.Env):
    '''
    Environment for longest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True, return_graph_obs=False, is_eval_env=False, parenting=-1) -> None:
        super(LongestPathEnv, self).__init__()
        
        assert parenting == -1, "Parenting not supported for LongestPathEnv"
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        
        self.EDGE_WEIGHT = 0
        
        self.n_nodes = n_nodes
        if n_edges == -1:
            n_edges = int((n_nodes * (n_nodes - 1) // 2) * 0.30)

        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)
        # self.observation_space = gym.spaces.Box(low=-1., high=11., shape=(2, n_nodes, n_nodes))        
        # self.observation_space = gym.spaces.Graph(
        #     node_space=gym.spaces.Box(low=0, high=4, shape=(1,)), 
        #     edge_space=gym.spaces.Box(low=0, high=1, shape=(1,)),
        #     )
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(2*n_nodes+2*n_edges+2*n_edges*2,))
        self.return_graph_obs = return_graph_obs
        self.solution_cost = 0
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
            d['delay'] = delay[u, v]
        
        G = G.to_directed()
        
        x = np.zeros((self.n_nodes, 2), dtype=np.float32)
        
        self.src, self.dest = np.random.choice(self.n_nodes, size=2, replace=False)
        x[self.src, self.NODE_HAS_MSG], x[self.dest, self.NODE_IS_TARGET] = 1, 1
        self.head = self.src
        self.adj = nx.adjacency_matrix(G, weight='delay').todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        self.optimal_solution = 0
        if self.is_eval_env:
            self.optimal_solution = -nx.shortest_path_length(G, source=self.src, target=self.dest, weight='delay')
            # self.optimal_solution = -1
        
        
        info = {'mask': self._get_mask()}
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        self.solution_cost = 0
        self.edges_taken = []
        

        return self._vectorize_graph(self.graph), info
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    
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
        reward = self.adj[self.head, action]
        self.solution_cost -= reward
        self.edges_taken.append((self.head, action))
        info = {}
        
        if np.isclose(self.graph.nodes[action, self.NODE_IS_TARGET], 1):
            done = True
            info['solved'] = True
        
        
        self.graph.nodes[action, self.NODE_HAS_MSG] = 1
        self.head = action
        
        info['mask'] = self._get_mask()
        if (not done) and (info['mask'].sum() == 0):
            done = True
            reward = -2*self.n_nodes
            info['solved'] = False
            
        if done:
            info['heuristic_solution'] = self.optimal_solution
            info['solution_cost'] = self.solution_cost
            info['edges_taken'] = self.edges_taken
            
        return self._vectorize_graph(self.graph), reward, done, False, info
        