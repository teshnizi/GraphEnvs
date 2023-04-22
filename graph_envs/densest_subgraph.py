import gymnasium as gym
import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random

class DensestSubgraphEnv(gym.Env):
    '''
    Environment for longest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, weighted=False, return_graph_obs=False, is_eval_env=False, parenting=-1) -> None:
        super(DensestSubgraphEnv, self).__init__()
        
        assert parenting in [0,1], "Parenting must be 0 or 1"
        assert weighted == False, "Weighted graphs not supported for this env"
        
        self.NODE_IS_TAKEN = 0
        
        self.EDGE_WEIGHT = 0
        
        self.n_nodes = n_nodes
        if n_edges == -1:
            n_edges = int((n_nodes * (n_nodes - 1) // 2) * 0.30)

        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.parenting = parenting
       
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(n_nodes+2*n_edges+2*n_edges*2,))
        self.return_graph_obs = return_graph_obs
        self.solution_cost = 0
        self.is_eval_env = is_eval_env

    def reset(self, seed=None, options={}) -> np.array:
        
        if seed != None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
        
        while True:
            G = nx.gnm_random_graph(self.n_nodes-1, self.n_edges)
            if nx.is_connected(G):
                break
        
        # separate node 0 from the rest of the graph:
        G.add_node(self.n_nodes-1)
        
        for v in G.nodes:
            G.nodes[v]['value'] = 1.0
            
        G.nodes[0]['value'] = 0.0
        G = G.to_directed()
        
        x = np.zeros((self.n_nodes, 1), dtype=np.float32)
        edge_f = np.ones((2*self.n_edges, 1), dtype=np.float32)
        edge_index = np.array(list(G.edges))
        
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        self.optimal_solution = 0
        if self.is_eval_env:
            # self.optimal_solution = -nx.shortest_path_length(G, source=self.src, target=self.dest, weight='delay')
            # self.optimal_solution = nx.large
            self.optimal_solution = -1
        
        info = {'mask': self._get_mask()}
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        
        self.solution_cost = 0
        self.nodes_taken = set()
        self.edge_taken_cnt = 0
        
        return self._vectorize_graph(self.graph), info
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        
        if self.graph.nodes[:, self.NODE_IS_TAKEN].sum() == 0:
            mask = np.zeros((self.n_nodes,), dtype=bool) < 1.0
            return mask
        
        if self.parenting == 0:
            mask = np.zeros((self.n_nodes,), dtype=bool) < 1.0
            mask[self.graph.nodes[:, self.NODE_IS_TAKEN] == 1] = 0
        
        elif self.parenting == 1:
            # print('==**==')
            # print(self.graph.nodes[:, self.NODE_IS_TAKEN])
            src_taken = self.graph.nodes[self.graph.edge_links[:,0], self.NODE_IS_TAKEN] == 1
            # print(src_taken)
            # print(self.graph.edge_links.T)
            valid_dests = self.graph.edge_links[src_taken, 1]
            # print(valid_dests.T)
            
            mask = np.zeros((self.n_nodes,), dtype=bool)
            mask[valid_dests] = 1
            mask[self.graph.nodes[:, self.NODE_IS_TAKEN] == 1] = 0
            # print(mask.T)
        # print('==**==')
        return mask
    
    def calculate_subgraph_density(self, sub_graph_nodes):
        sub_graph = nx.subgraph(self.G, sub_graph_nodes)
        return sub_graph.number_of_edges() / sub_graph.number_of_nodes()
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        # assert action in self._get_neighbors(self.head), f"Node {action} is not a neighbor of the current path head {self.head}!"
        assert np.isclose(self.graph.nodes[action, self.NODE_IS_TAKEN], 1) == False, f"Node {action} is already a part of the path!"
        
        
        done = False
        info = {}
        info['heuristic_solution'] = self.optimal_solution
        info['solved'] = True
        
        if action == self.n_nodes-1:
            reward = 0
            done = True
            info['nodes_taken'] = self.nodes_taken
            info['solution_cost'] = self.solution_cost
            info['mask'] = self._get_mask()
            return self._vectorize_graph(self.graph), reward, done, False, info
        
        new_edges = 0
        
        for v in self._get_neighbors(action):
            if v in self.nodes_taken:
                new_edges += 1
        
        
        if len(self.nodes_taken) == 0:
            reward = 0
        else:
            reward = ((self.edge_taken_cnt + new_edges) / (len(self.nodes_taken)+1)) - (self.edge_taken_cnt / len(self.nodes_taken))
        
        # print('Edges taken:', self.edge_taken_cnt, 'Nodes taken:', len(self.nodes_taken))
        # print('Action:', action, 'Reward:', reward, 'New edges:', new_edges, 'Solution cost:', self.solution_cost)
        
        self.edge_taken_cnt += new_edges
        self.nodes_taken.add(action)
        self.graph.nodes[action, self.NODE_IS_TAKEN] = 1
        
        self.solution_cost = self.edge_taken_cnt / len(self.nodes_taken)
                
        
        info['mask'] = self._get_mask()

            
        if (info['mask'].sum() == 0):
            assert len(self.nodes_taken) == self.n_nodes-1, "Bug!"
            assert self.edge_taken_cnt == self.n_edges, "Bug!"
            assert (self.graph.nodes[:-1, self.NODE_IS_TAKEN]==1).all(), "Bug!"
            reward = 0
            done = True
            
        if done:
            info['nodes_taken'] = self.nodes_taken
            info['solution_cost'] = self.solution_cost
            
        return self._vectorize_graph(self.graph), reward, done, False, info
        