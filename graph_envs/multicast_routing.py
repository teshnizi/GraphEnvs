import gymnasium as gym
# import torch 
# import torch_geometric as pyg 
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import networkx as nx 
import random

from typing import Tuple
from matplotlib import pyplot as plt

class MulticastRoutingEnv(gym.Env):
    '''
    Environment for multicast routing problem.
    '''
    # Meta Data:
    n_node_features = 4
    n_edge_features = 2
    has_mask = True
    
    def get_mask_shape(self):
        return (self.n_edges,)
    
    def __init__(self, n_nodes, n_edges=-1, n_dests=3, weighted=True, max_distance=-1, parenting=4, is_eval_env=False) -> None:
        super(MulticastRoutingEnv, self).__init__()
        
        if parenting not in [1,2,3,4]:
            raise ValueError('Invalid parenting type')
        self.parenting = parenting
        
        
        # Node status codes
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        self.NODE_MAX_DISTANCE = 2
        self.NODE_DISTANCE_FROM_SOURCE = 3
    
        
        # Edge status codes    
        self.EDGE_DELAY = 0
        self.EDGE_IS_TAKEN = 1
        
        
        self.n_nodes = n_nodes
        if n_edges == -1:
            n_edges = int((n_nodes * (n_nodes - 1) // 2) * 0.30)

        self.n_edges = n_edges
        self.n_dests = n_dests
        self.weighted = weighted
        self.parenting = parenting
        
        #TODO: This should be a parameter
        if max_distance == -1:
            max_distance = np.log(n_nodes) * (1+0.3)/2
            
        # self.max_distance = max_distance
            
        self.action_space = gym.spaces.Discrete(n_edges)
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(self.n_node_features*n_nodes+self.n_edge_features*n_edges*2+2*n_edges*2,))
    
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
        
        
        x = np.zeros((self.n_nodes, self.n_node_features), dtype=np.float32)      
        
        self.src = 0
        self.dests = np.random.choice(np.arange(1, self.n_nodes), size=self.n_dests, replace=False)
        # src, self.dests = self.dests[0], self.dests[1:]
        
        shortest_paths = nx.shortest_path_length(G, source=self.src, weight='delay')
        farthest_target = max([shortest_paths[d] for d in self.dests])
        farthest_node = max(shortest_paths.values())
        
        sum_of_shortest_paths = sum([shortest_paths[d] for d in self.dests])
        max_distance = np.random.rand() * (farthest_node - farthest_target) + farthest_target
        
        self.approx_solution = 0
        
        if self.is_eval_env:
            shortest_paths = nx.shortest_path(G, source=self.src, weight='delay')
            all_path_edges = set()
            for d in self.dests:
                for u, v in zip(shortest_paths[d][:-1], shortest_paths[d][1:]):
                    assert (v, u) not in all_path_edges, "Wrong!"
                    all_path_edges.add((u, v))
            
            self.approx_solution = sum([G[u][v]['delay'] for u, v in all_path_edges])
        
        
        G = G.to_directed()
        self.G = G
        
        # A tree that's used to keep track of distance of each node from the source given the partial tree
        if self.parenting == 4:
            self.alt_G = G.copy()
        
        x[self.src, self.NODE_HAS_MSG] = 1
        x[self.dests, self.NODE_IS_TARGET] = 1
        # x[:, self.NODE_MAX_DISTANCE] = self.max_distance
        x[:, self.NODE_MAX_DISTANCE] = max_distance
        x[:, self.NODE_DISTANCE_FROM_SOURCE] = -1
        x[self.src, self.NODE_DISTANCE_FROM_SOURCE] = 0
        
                
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
       
        edge_f = np.concatenate((edge_f, np.zeros((2*self.n_edges, 1), dtype=np.float32)), axis=1)
        
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        self.generated_solution = []

        self.solution_cost = 0
        self.constraints_satisfied = 0
        
        info = {'mask': self._get_mask()}
        self._vectorize_graph(self.graph)
        

        return self._vectorize_graph(self.graph), info
    
    
    def _vectorize_graph(self, graph):
        return np.concatenate((graph.nodes.flatten(), graph.edges.flatten(), graph.edge_links.flatten()), dtype=np.float32)
    
    def _get_mask(self) -> np.array:
        # print('=========')
        mask = np.zeros((2 * self.n_edges,), dtype=bool) < 1.0

        mask[self.graph.nodes[self.graph.edge_links[:, 0], self.NODE_HAS_MSG] < 0.5] = False
        mask[self.graph.nodes[self.graph.edge_links[:, 1], self.NODE_HAS_MSG] > 0.5] = False
        
        # print('Pre: ', mask.nonzero()[0])
        if self.parenting >= 3:
            Vs = self.graph.edge_links[mask, 1]
            Vs = np.unique(Vs)
            mask[:] = False
            
            s_to_Vs = [-1] * self.n_nodes

            for id, v in enumerate(Vs):
                neighbor_edges = (self.graph.edge_links[:, 1] == v)
                edges_to_tree = neighbor_edges & (self.graph.nodes[self.graph.edge_links[:, 0], self.NODE_HAS_MSG] > 0.5)
                distances = self.graph.nodes[self.graph.edge_links[edges_to_tree, 0], self.NODE_DISTANCE_FROM_SOURCE] \
                    + self.graph.edges[edges_to_tree, self.EDGE_DELAY]    
                    
                all_distances = self.graph.nodes[self.graph.edge_links[:, 0], self.NODE_DISTANCE_FROM_SOURCE] \
                    + self.graph.edges[:, self.EDGE_DELAY]
                all_distances[edges_to_tree == False] = np.inf
                
                s_to_Vs[v] = np.min(distances)
                best_edge = np.argmin(all_distances)
                mask[best_edge] = True
                # print(f'V: {v}, Best edge: {self.graph.edge_links[best_edge]}')
            
        return mask
    
    
    def step(self, action: Tuple[int, int]) -> Tuple[gym.spaces.GraphInstance, SupportsFloat, bool, bool, dict]:
        
        u, v = self.graph.edge_links[action, 0], self.graph.edge_links[action, 1]
        
        # print(f'Action: {action}, u: {u}, v: {v}')
        assert (u < self.n_nodes), f"Node {u} is out of bounds!"
        assert (v < self.n_nodes), f"Node {v} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert np.isclose(self.graph.edges[action, self.EDGE_IS_TAKEN], 1) == False, f"BUG IN THE LIBRARY! Edge {action} is already a part of the path!"
        assert np.isclose(self.graph.nodes[u, self.NODE_HAS_MSG], 1), f"BUG IN THE LIBRARY! Node {u} is not a part of the path!"
        assert np.isclose(self.graph.nodes[v, self.NODE_HAS_MSG], 0), f"BUG IN THE LIBRARY! Node {v} is already a part of the path!"
        
        done = False
        reward = -self.graph.edges[action, self.EDGE_DELAY]
        info = {}
        
        self.solution_cost -= reward
        self.graph.nodes[v, self.NODE_HAS_MSG] = 1
        self.graph.edges[action, self.EDGE_IS_TAKEN] = 1
        
        self.generated_solution.append((u, v))
        
        self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] = \
            self.graph.nodes[u, self.NODE_DISTANCE_FROM_SOURCE]\
                + self.graph.edges[action, self.EDGE_DELAY]
        
        if self.graph.nodes[v, self.NODE_IS_TARGET] == 1:
            if self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] > self.graph.nodes[v, self.NODE_MAX_DISTANCE] + 1e-4:
                reward -= 2*self.n_nodes
            else:
                reward += 1
                self.constraints_satisfied += 1
        
        
        dests_left = np.logical_and(self.graph.nodes[:, self.NODE_HAS_MSG] < 1e-5, self.graph.nodes[:, self.NODE_IS_TARGET] > (1-1e-5) ).sum()
        
        
        if self.parenting == 4:
            
            neighbors_of_v = list(self.alt_G.neighbors(v))
            all_edges = set(self.alt_G.edges())
            
            # print(all_edges)
            # print(f'V: {v}, Neighbors: {neighbors_of_v}')
            assert 0 in neighbors_of_v, "BUG IN THE LIBRARY! Node 0 is not a neighbor of v!"
            for w in neighbors_of_v:
                if w == 0:
                    continue
                assert self.graph.nodes[w, self.NODE_HAS_MSG] < 0.5, "BUG IN THE LIBRARY! Node w is already a part of the tree!"
                # print(f'W: {w}, Edge: {(0, w)}')
                
                if (0, w) in all_edges:
                    self.alt_G.edges[0, w]['delay'] = np.minimum(\
                        self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] + self.alt_G.edges[v, w]['delay'],
                        self.alt_G.edges[0, w]['delay'])
                    self.alt_G.edges[w, 0]['delay'] = self.alt_G.edges[0, w]['delay']
                    
                    # print('Replace ', self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE], self.alt_G.edges[v, w]['delay'], self.alt_G.edges[0, w]['delay'])
                else:
                    # print('Make    ', self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE], self.alt_G.edges[v, w]['delay'])
                    self.alt_G.add_edge(0, w, delay=self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] + self.alt_G.edges[v, w]['delay'])
                    self.alt_G.add_edge(w, 0, delay=self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] + self.alt_G.edges[v, w]['delay'])
            
            # print('Done')
            self.alt_G.remove_node(v)
            
            # print('===== Alt Graph =====')
            # print('Nodes: ', self.alt_G.nodes.data())
            # print('Edges: ', self.alt_G.edges.data())
            
            # draw_g = self.alt_G.copy()
            # for e in draw_g.edges:
            #     # print(G.edges[e], G.edges[e]['edge_attr'])
            #     draw_g.edges[e]['delay'] = round(draw_g.edges[e]['delay'] * 10)/10
            
            # plt.figure(figsize=(10, 10))
            # layout = nx.kamada_kawai_layout(draw_g)
            # nx.draw(draw_g, pos=layout, with_labels=True)
            # nx.draw_networkx_edge_labels(draw_g, pos=layout, edge_labels=nx.get_edge_attributes(draw_g, 'delay'), font_color='red')
            # plt.savefig(f"graph_{v}.png")
            
            sps = nx.shortest_path_length(self.alt_G, 0, weight='delay')
            # print('SPS: ', sps)
            for d in self.dests:
                if d in self.alt_G.nodes:
                    if sps[d] > self.graph.nodes[d, self.NODE_MAX_DISTANCE] + 1e-5:
                        # print('HELPED!')
                        reward -= 2*self.n_nodes * dests_left
                        done = True
                        info['solved'] = False
                        break
            
        info['mask'] = self._get_mask()
        
        
        if dests_left == 0:
            done = True
            if self.constraints_satisfied == self.n_dests:
                info['solved'] = True
            else:
                info['solved'] = False
                
        elif info['mask'].sum() == 0:
            if self.parenting <= 1:
                assert info['mask'].sum() > 0, "No more actions possible! Shouldn't happen!"
            else:
                reward -= 2*self.n_nodes * dests_left
                done = True
                info['solved'] = False
            
            
        if done:
            info['heuristic_solution'] = self.approx_solution
            info['solution_cost'] = self.solution_cost
        
        
        return self._vectorize_graph(self.graph), reward, done, False, info
