import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random
from graph_envs.utils import vectorize_graph
import graph_envs.feature_extraction as fe

class TSPEnv(gym.Env):
    '''
    Environment for shortest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=True, return_graph_obs=False, parenting=-1, is_eval_env=False) -> None:
        super(TSPEnv, self).__init__()
        
        assert parenting in [1,2], "Parenting must be either 1 or 2"
        
        
        self.NODE_TAKEN = 0
        self.NODE_IS_SOURCE = 1
        self.EDGE_WEIGHT = 0
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.weighted = weighted
        
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Box(low=-10, high=1000, shape=((2 + fe.get_num_features()) *n_nodes+2*n_edges+2*n_edges*2,))
        
        self.return_graph_obs = return_graph_obs
        self.is_eval_env = is_eval_env
        self.parenting = parenting
        
        
    def reset(self, seed=None, options={}) -> np.array:
        
        if seed != None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
        
        self.start = 0
        
        
        while True:
            G = nx.gnm_random_graph(self.n_nodes, self.n_edges)
            if not nx.is_connected(G):
                continue
            
            if any([G.degree(node) == 1 for node in G.nodes]):
                continue
            
            cp = G.copy()
            cp.remove_node(self.start)
            if nx.is_connected(cp):
                break
        
        if self.weighted:
            delay = np.random.randint(3, 10, size=(self.n_nodes, self.n_nodes))/10.0
        else:
            delay = np.random.randint(1, 2, size=(self.n_nodes, self.n_nodes))/1.0
            
        
        
        for u, v, d in G.edges(data=True):
            d['weight'] = delay[u, v]
        
        self.optimal_solution = 0
        
        if self.is_eval_env == True:
            self.optimal_cycle = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)
            for i in range(len(self.optimal_cycle)-1):
                self.optimal_solution += G[self.optimal_cycle[i]][self.optimal_cycle[i+1]]['weight']
            
       
       
        if self.parenting >= 2:
            self.alt_G = G.copy()
            self.alt_G.remove_node(self.start)
            
            
        G = G.to_directed()
        self.G = G
        
        x = np.zeros((self.n_nodes, 2+fe.get_num_features()), dtype=np.float32)
        
        
        x[self.start, self.NODE_IS_SOURCE] = 1
        self.head = self.start
            
        # Adding structural features
        sf = fe.generate_features(G)
        x[:, -fe.get_num_features():] = sf
        
        self.adj = nx.adjacency_matrix(G, weight='weight').todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['weight'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
         
        self.total_solution_cost = 0
        self.steps_taken = 0
        
        info = {'mask': self._get_mask()}
        
        # assert info['mask'].sum() > 0, "No valid actions!"
        if info['mask'].sum() == 0:
            info['mask'][self.start] = 1
            
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        

        return vectorize_graph(self.graph), info
    
    # def _vectorize_graph(self, graph):
    #     return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((self.n_nodes,), dtype=bool)
        mask[self._get_neighbors(self.head)] = 1
        mask[self.graph.nodes[:, self.NODE_TAKEN] == 1] = 0
        if self.graph.nodes[:, self.NODE_TAKEN].sum() < self.n_nodes - 1:
            mask[self.start] = 0
            
        if self.parenting >= 2:
            valid_nodes = (mask == 1).nonzero()[0]
            
            # disconnecting_nodes = []
            
            for v in valid_nodes:
                if v == self.start:
                    continue
                G_copy = self.alt_G.copy()
                G_copy.remove_node(v)
                if G_copy.number_of_nodes() == 0:
                    break
                if not nx.is_connected(G_copy):
                    mask[v] = 0
                    
        
        

        return mask
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        if (action == self.start) and (self.head == self.start):
            done = True
            reward = - self.n_nodes
            info = {}
            info['solved'] = False
            info['heuristic_solution'] = self.optimal_solution
            info['solution_cost'] = -1
            info['mask'] = self._get_mask()
            return vectorize_graph(self.graph), reward, done, False, info
            
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert action in self._get_neighbors(self.head), f"Node {action} is not a neighbor of the current path head {self.head}!"
        assert self.graph.nodes[action, self.NODE_TAKEN] == 0, f"Node {action} is already taken!"
        
        self.steps_taken += 1
        
        info = {}
        done = False
      
        
        # Calculate the reward:
        reward = 0
        reward = reward - self.adj[self.head, action]
        # reward = reward + 1 - self.graph.nodes[action, self.NODE_TAKEN]
        
        # Add the cost of the edge to the total solution cost:
        self.total_solution_cost += self.adj[self.head, action]
        
        # Mark the node as taken:
        self.graph.nodes[action, self.NODE_TAKEN] = 1
        
        if self.parenting >= 2:
            if action != self.start:
                self.alt_G.remove_node(action)
            
        # Update the head:
        self.head = action
        
        # If all nodes are taken and the head is back to the start node, the episode is done:
        if np.isclose(self.graph.nodes[:, self.NODE_TAKEN], 0).any() == False:   
            if action == self.start: 
                done = True
                info['solved'] = True
        
        info['mask'] = self._get_mask()
        if (not done) and (info['mask'].sum() == 0):
            done = True
            reward -= self.n_nodes * 2
            info['solved'] = False
            
        if done:
            info['heuristic_solution'] = self.optimal_solution
            info['solution_cost'] = self.total_solution_cost
           
           
        return vectorize_graph(self.graph), reward, done, False, info
        