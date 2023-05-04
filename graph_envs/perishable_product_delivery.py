import gymnasium as gym
import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Tuple

import networkx as nx 
import random


from graph_envs.utils import vectorize_graph
import graph_envs.feature_extraction as fe



class PerishableProductDeliveryEnv(gym.Env):
    '''
    Environment for longest path problem.
    input:
        - n_nodes: number of nodes in the graph
        - n_edges: number of edges in the graph
        - weighted: whether the graph is weighted or not
    '''
    
    def __init__(self, n_nodes, n_edges, n_products=3, delivery_time=-1, weighted=True, return_graph_obs=False, is_eval_env=False, parenting=-1) -> None:
        super(PerishableProductDeliveryEnv, self).__init__()
        
        assert parenting in [1], "Parenting must be 1!"
        # 0: no parenting
        # 1: Removing nodes that are not connected to the head
        # 2: Removing nodes that result in time exceeding the delivery time
        
        assert n_products <= 5, "Max 5 products!"
        
        self.NODE_IS_HEAD = 0
        self.NODE_HAS_P = np.array([1, 2, 3, 4, 5])
        self.NODE_NEEDS_P = np.array([6, 7, 8, 9, 10])
        self.NODE_TIME_LEFT_P = np.array([11, 12, 13, 14, 15])
        
        
        self.EDGE_WEIGHT = 0
        
        self.max_products = len(self.NODE_HAS_P)
        self.n_products = n_products
        
        self.n_nodes = n_nodes
        if n_edges == -1:
            n_edges = int((n_nodes * (n_nodes - 1) // 2) * 0.30)

        if delivery_time == -1:
            avg_degree = 2*n_edges/n_nodes
            avg_dist = np.log(n_nodes)/np.log(avg_degree)
            if weighted:
                avg_dist = avg_dist * (0.3+1.0)/2.0
                
            self.dt_mn = avg_dist*0.6
            self.dt_mx = avg_dist*1.4
            
        
        self.delivery_time = -1
        self.n_edges = n_edges
        self.weighted = weighted
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.parenting = parenting
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=((1+3*self.max_products+fe.get_num_features())*n_nodes+2*n_edges+2*n_edges*2,))
        self.return_graph_obs = return_graph_obs
        self.solution_cost = 0
        self.is_eval_env = is_eval_env
        self.max_steps = self.n_nodes * self.n_products * 50

    def reset(self, seed=None, options={}) -> np.array:
        
        
        if seed != None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
        
        self.pickups = [-1]*self.n_products
        self.dropoffs = [-1]*self.n_products
        
        while True:
            G = nx.gnm_random_graph(self.n_nodes, self.n_edges)
            
            if not nx.is_connected(G):
                continue
            
            if self.weighted:
                delay = np.random.randint(3, 10, size=(self.n_nodes, self.n_nodes))/10.0
            else:
                delay = np.random.randint(10, 11, size=(self.n_nodes, self.n_nodes))/10.0
                
            
            for u, v, d in G.edges(data=True):
                d['delay'] = delay[u, v]
            
            self.delivery_time = np.random.rand()*(self.dt_mx-self.dt_mn) + self.dt_mn
            
            apsp = nx.floyd_warshall(G, weight='delay')
            
            for i in range(self.n_products):
                # choose a pickup from nodes not already chosen
                self.pickups[i] =  np.random.choice([node for node in range(self.n_nodes) \
                    if (not node in self.pickups) and (not node in self.dropoffs)])
                close_nodes = [node for node in apsp[self.pickups[i]] \
                    if (apsp[self.pickups[i]][node] < self.delivery_time + 1e-6)\
                        and (not node in self.pickups) and (not node in self.dropoffs)]
                if len(close_nodes) == 0:
                    break
                self.dropoffs[i] = np.random.choice(close_nodes)
            
            if (not -1 in self.pickups) and (not -1 in self.dropoffs):
                break
        
        
        self.G = G
        G = G.to_directed()
        self.apsp = apsp
        
        x = np.zeros((self.n_nodes, 1 + 3*self.max_products + fe.get_num_features()), dtype=np.float32)
        
        for i in range(self.n_products):
            x[self.pickups[i], self.NODE_HAS_P[i]] = 1
            x[self.dropoffs[i], self.NODE_NEEDS_P[i]] = 1
            x[:, self.NODE_TIME_LEFT_P[i]] = self.delivery_time
                

        # Adding structural features
        sf = fe.generate_features(G)
        x[:, -fe.get_num_features():] = sf
        
                
        self.head = 0
        x[self.head, self.NODE_IS_HEAD] = 1
        
        self.adj = nx.adjacency_matrix(G, weight='delay').todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        
        self.optimal_solution = 0
        if self.is_eval_env:
            total_length = 0
            curr_node = self.head
            for product in range(self.n_products):
                total_length += nx.shortest_path_length(G, source=curr_node, target=self.pickups[product], weight='delay')
                total_length += nx.shortest_path_length(G, source=self.pickups[product], target=self.dropoffs[product], weight='delay')
            self.optimal_solution = total_length
        
        info = {'mask': self._get_mask()}
        
        if self.return_graph_obs:
            info['graph_obs'] = self.graph
        info['pickups'] = self.pickups
        info['dropoffs'] = self.dropoffs
        info['time_left'] = self.delivery_time
        
        self.solution_cost = 0
        self.edges_taken = []


        return vectorize_graph(self.graph), info
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_neighbors(self, node):
        neigbors = self.graph.edge_links[self.graph.edge_links[:, 0] == node, 1]
        return neigbors
        
    def _get_mask(self) -> np.array:
        mask = np.zeros((self.n_nodes,), dtype=bool)
        for i in range(self.n_products):
            if self.graph.nodes[self.head, self.NODE_HAS_P[i]] == 1:
                mask[self.head] = True
                
        mask[self._get_neighbors(self.head)] = True
        
        if self.parenting >= 2:
            # For each product already taken, mask nodes that are farther from the dropoff than the remaining delivery time:
            valid_nodes = self._get_neighbors(self.head)
            for i in range(self.n_products):
                if self.graph.nodes[self.head, self.NODE_HAS_P[i]] == -1:
                    for v in valid_nodes:
                        if self.apsp[v][self.dropoffs[i]] + self.adj[self.head, v] > self.graph.nodes[self.head, self.NODE_TIME_LEFT_P[i]]:
                            mask[v] = False
                            # print(f'Node {v} masked because of product {i} with time left {self.graph.nodes[v, self.NODE_TIME_LEFT_P[i]]}!')
                    
                
        return mask
    
    def step(self, action: int) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        assert (action < self.n_nodes), f"Node {action} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"

        if self.parenting >= 1:
            assert action in self._get_neighbors(self.head) or action==self.head, f"Node {action} is an invalid action with head {self.head}!"
            
        
        done = False
        reward = 0
        
        info = {}
        info['heuristic_solution'] = self.optimal_solution
        info['solution_cost'] = self.solution_cost
        info['edges_taken'] = self.edges_taken
            
        if action == self.head:
            # If the head is the same as the action, then the agent is picking up a product
            assert (self.graph.nodes[self.head, self.NODE_HAS_P] == 0).sum() > 0, "No products to pick up!"
            prod = np.random.choice(np.where(self.graph.nodes[self.head, self.NODE_HAS_P] == 1)[0])
            self.graph.nodes[:, self.NODE_HAS_P[prod]] = -1
            
            reward += 2
            self.edges_taken.append((self.head, self.head))
            
        else:
            # If the head is not the same as the action, then the agent is moving to a new node
            reward = -self.adj[self.head, action]
            self.solution_cost -= reward
            self.edges_taken.append((self.head, action))
            
            self.graph.nodes[self.head, self.NODE_IS_HEAD] = 0
            self.graph.nodes[action, self.NODE_IS_HEAD] = 1
            self.head = action
            
            for prod in range(self.n_products):
                # If the product is taken, take the edge and reduce the product time left
                if self.graph.nodes[self.head, self.NODE_HAS_P[prod]] == -1:
                    self.graph.nodes[:, self.NODE_TIME_LEFT_P[prod]] -= self.adj[self.head, action]
                    # print(f'Product {prod} time left: {self.graph.nodes[:, self.NODE_TIME_LEFT_P[prod]]}')
                    if self.graph.nodes[:, self.NODE_TIME_LEFT_P[prod]].sum() < 0 - 1e-6:
        
                        done = True
                        reward = -2*self.n_nodes*self.n_products
                        info['solved'] = False
                        return vectorize_graph(self.graph), reward, done, False, info
                    
                    if self.graph.nodes[self.head, self.NODE_NEEDS_P[prod]] == 1:
                        # If the product is needed, check if it is delivered. If so, remove the product
        
                        reward += 2
                        self.graph.nodes[:, self.NODE_HAS_P[prod]] = 0
                        self.graph.nodes[:, self.NODE_NEEDS_P[prod]] = 0
                        self.graph.nodes[:, self.NODE_TIME_LEFT_P[prod]] = 0
            
        
        if self.graph.nodes[:, self.NODE_HAS_P].sum() == 0:
            done = True
            info['solved'] = True
            reward += 2*self.n_nodes
        elif len(self.edges_taken) >= self.max_steps:
            done = True
            info['solved'] = False
            reward = -2*self.n_nodes*self.n_products
            
        info['mask'] = self._get_mask()       
        
        if self.parenting <= 1:
            assert (info['mask'].sum() > 0), f"Mask is empty!"
        elif self.parenting == 2:
            if info['mask'].sum() == 0:
                done = True
                info['solved'] = False
                reward = -2*self.n_nodes*self.n_products

            
        return vectorize_graph(self.graph), reward, done, False, info
        