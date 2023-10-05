import gymnasium as gym
import netgen
import numpy as np 

from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import networkx as nx 
import random

from typing import Tuple
from matplotlib import pyplot as plt


from graph_envs.utils import vectorize_graph
import graph_envs.feature_extraction as fe


class MulticastRoutingEnv(gym.Env):
    '''
    Environment for multicast routing problem.
    '''
    # Meta Data:
    # n_node_features = 4
    n_edge_features = 2
    has_mask = True
    
    def get_mask_shape(self):
        return (self.n_edges,)
    
    # def __init__(self, n_nodes, n_edges=-1, n_dests=3, weighted=True, max_distance=-1, parenting=4, is_eval_env=False) -> None:
    def __init__(self, config, is_eval_env=False) -> None:
        super(MulticastRoutingEnv, self).__init__()
        
        parenting = config.parenting

        # This lets you choose the parenting level or strategy. 1 is no parenting. Use 2 until the agent works, and then you can try the other ones.
        
        if parenting not in [1,2,3,4]:
            raise ValueError('Invalid parenting type')
        self.parenting = parenting


        # Status codes are used to show node/edge states at each state of the whole graph. 
        
        # Node status codes. 
        self.NODE_HAS_MSG = 0
        self.NODE_IS_TARGET = 1
        self.NODE_MAX_DISTANCE = 2
        self.NODE_DISTANCE_FROM_SOURCE = 3
    
        
        # Edge status codes    
        self.EDGE_DELAY = 0
        self.EDGE_IS_TAKEN = 1
        

        # These values show number of nodes and edges
        self.n_nodes = config.num_nodes
        self.n_edges = config.num_arcs
        
        # self.n_dests = config.?    # EXTRACT FROM CONFIG!
        # self.weighted = True
        # self.parenting = config.parenting
        
    
        # self.max_distance = config.?    # EXTRACT FROM CONFIG!
        '''
         These two lines define the action space and observation space in the reinforcement learning model. Each action is taking an edge, 
         so we have self.n_edges actions (some of them might be masked depending on the state. 
         
         Each observations shows is a flattened vector incorporating all the information necessary to uniquely identify and represent a state. 
         We flatten the observations because of technical reasons (the other RL libraries don't work with graph observations)
         
         the graph state:
         - 4 is the number of problem-specific features for each node: 
             0: NODE_HAS_MSG,
             1: NODE_IS_TARGET,
             2: NODE_MAX_DISTANCE, 
             3: NODE_DISTANCE_FROM_SOURCE
         - fe.get_num_features() is the number of structural features added to each node.
         For instance in each node's feature vector, element 0 is either 0 or 1 and shows whether that node has access to the message or not. 
         
         - self.n_edge_features is equal to 2 and shows the number of edge features:
             0: EDGE_DELAY
             1: EDGE_IS_TAKEN
        For instance in each edge's feature vector, element 0 is a floating point number representing the delay of the edge.
         
         the "*2" in "self.n_edge_features*self.n_edges*2" is because we have an undirected graph, and we use two directed edges (one in each direction) for each edge.
         finally, the "2*self.n_edges*2" features at the end of the observation vector is for showing the edges themselves (it's flattened edge_index)
         
        '''
        self.action_space = gym.spaces.Discrete(self.n_edges)
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, \
            shape=((4 +fe.get_num_features()) * self.n_nodes+self.n_edge_features*self.n_edges*2+2*self.n_edges*2,))
    
        self.is_eval_env = is_eval_env

                
    def reset(self, seed=None, options={}) -> np.array:
        if seed != None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)

        # generating a random connected graph 
        while True:
            # G = nx.gnm_random_graph(self.n_nodes, self.n_edges)
            graph = netgen.sp3randGenTE(
                self.config.seed,
                self.config.num_nodes,
                self.config.num_arcs,
                self.config.num_provisioned_bundles,
                self.config.num_all_bundles,
                self.config.num_priorities,
                self.config.bandwidth_lb,
                self.config.bandwidth_ub,
                self.config.latency_lb,
                self.config.latency_ub,
                self.config.rate_lb,
                self.config.rate2_lb,
                self.config.rate_ub,
                self.config.rate2_ub,
                self.config.delay_lb,
                self.config.delay2_lb,
                self.config.delay_ub,
                self.config.delay2_ub,
                self.config.num_unicast_reqs_per_bundle_lb,
                self.config.num_unicast_reqs_per_bundle_ub,
                self.config.bundle_size_lb,
                self.config.bundle_size_ub,
                self.config.request_size_lb,
                self.config.request_size_ub,
                False,  # Don't allow multiple edges
            )
            
            # ==============================================
            # TODO: Convert netgen graph into networkx graph
            # ==============================================
            
            if nx.is_connected(G):
                break

        # sampling weights for the edges
        if self.weighted:
            delay = np.random.randint(3, 10, size=(self.n_nodes, self.n_nodes))/10.0
        else:
            delay = np.random.randint(1, 2, size=(self.n_nodes, self.n_nodes))/1.0
        
        for u, v, d in G.edges(data=True):
            d['delay'] = delay[u, v]
        
        x = np.zeros((self.n_nodes, 4 + fe.get_num_features()), dtype=np.float32)      
        
        self.src = 0

        # sampling the destinations
        self.dests = np.random.choice(np.arange(1, self.n_nodes), size=self.n_dests, replace=False)
        

        # finding shortest paths to use as a heuristic (sum of lengths of all shortest paths from SRC to destinations is an upperbound for the actual solution)
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

        # setting the graph features 
        x[self.src, self.NODE_HAS_MSG] = 1
        x[self.dests, self.NODE_IS_TARGET] = 1
        # x[:, self.NODE_MAX_DISTANCE] = self.max_distance
        x[:, self.NODE_MAX_DISTANCE] = max_distance
        x[:, self.NODE_DISTANCE_FROM_SOURCE] = -1
        x[self.src, self.NODE_DISTANCE_FROM_SOURCE] = 0
        
        # Adding structural features
        sf = fe.generate_features(G)
        x[:, -fe.get_num_features():] = sf
                
        self.adj = nx.adjacency_matrix(G).todense()
        
        edge_index = np.array(list(G.edges))
        edge_f = np.array([G[u][v]['delay'] for u, v in G.edges], dtype=np.float32).reshape(-1, 1)
       
        edge_f = np.concatenate((edge_f, np.zeros((2*self.n_edges, 1), dtype=np.float32)), axis=1)

        # here we define the graph
        self.graph = gym.spaces.GraphInstance(nodes=x, edges=edge_f, edge_links=edge_index)
        self.generated_solution = []
        
        self.solution_cost = 0
        self.constraints_satisfied = 0
        
        info = {'mask': self._get_mask()}

        return vectorize_graph(self.graph), info
    
    
   
    def _get_mask(self) -> np.array:
        '''
        This function generates a mask for the current state. The mask is binary vector of the same shape as action space (self.n_edges), and shows whether each 
        edge can be taken and added to the partial solution or not.
        '''
        mask = np.zeros((2 * self.n_edges,), dtype=bool) < 1.0

        # maskign the eges that are already taken
        mask[self.graph.edges[:, self.EDGE_IS_TAKEN] > 0.5] = False
        
        if self.parenting >= 2:
            # masking the edges that connect the head to the nodes that already have the message
            mask[self.graph.nodes[self.graph.edge_links[:, 0], self.NODE_HAS_MSG] < 0.5] = False
            mask[self.graph.nodes[self.graph.edge_links[:, 1], self.NODE_HAS_MSG] > 0.5] = False
        
    
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
        '''
        This function adds an edge to the partial solution (takes an action in the RL framework)
        '''
        u, v = self.graph.edge_links[action, 0], self.graph.edge_links[action, 1]
        
        # making sure the action is valid
        assert (u < self.n_nodes), f"Node {u} is out of bounds!"
        assert (v < self.n_nodes), f"Node {v} is out of bounds!"
        assert self._get_mask()[action] == True, f"Mask of {action} is False!"
        assert np.isclose(self.graph.edges[action, self.EDGE_IS_TAKEN], 1) == False, f"BUG IN THE LIBRARY! Edge {action} is already a part of the path!"
        
        if self.parenting >= 2:
            assert np.isclose(self.graph.nodes[u, self.NODE_HAS_MSG], 1), f"BUG IN THE LIBRARY! Node {u} is not a part of the path!"
            assert np.isclose(self.graph.nodes[v, self.NODE_HAS_MSG], 0), f"BUG IN THE LIBRARY! Node {v} is already a part of the path!"
        
        done = False
        reward = -self.graph.edges[action, self.EDGE_DELAY]
        info = {}
        info['heuristic_solution'] = self.approx_solution
        info['solution_cost'] = -1
        
        self.solution_cost -= reward

        # if the edge is creating a cycle
        if np.isclose(self.graph.nodes[u, self.NODE_HAS_MSG], 0) or np.isclose(self.graph.nodes[v, self.NODE_HAS_MSG], 1):
            assert self.parenting <= 1, f"Wrong parenting!"
            reward = -2*self.n_nodes * self.n_dests
            done = True
            info['solved'] = False
            info['mask'] = self._get_mask()
            return vectorize_graph(self.graph), reward, done, False, info
            
        # update the state
        self.graph.nodes[v, self.NODE_HAS_MSG] = 1
        self.graph.edges[action, self.EDGE_IS_TAKEN] = 1
        
        self.generated_solution.append((u, v))

        # update the distance from source in the current tree for node v 
        self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] = \
            self.graph.nodes[u, self.NODE_DISTANCE_FROM_SOURCE]\
                + self.graph.edges[action, self.EDGE_DELAY]

        # if the edge connects the source to a new destination
        if self.graph.nodes[v, self.NODE_IS_TARGET] == 1:
            if self.graph.nodes[v, self.NODE_DISTANCE_FROM_SOURCE] > self.graph.nodes[v, self.NODE_MAX_DISTANCE] + 1e-4:
                reward = -2*self.n_nodes * self.n_dests
                done = True
                info['solved'] = False
                info['mask'] = self._get_mask()
                return vectorize_graph(self.graph), reward, done, False, info
            
            reward += 1
            self.constraints_satisfied += 1
    
        
        dests_left = np.logical_and(self.graph.nodes[:, self.NODE_HAS_MSG] < 1e-5, self.graph.nodes[:, self.NODE_IS_TARGET] > (1-1e-5) ).sum()
        info['mask'] = self._get_mask()
        
        if self.parenting <= 1:
            assert info['mask'].sum() > 0, "No more actions possible! Shouldn't happen!"

        # if all destinations are reached
        if dests_left == 0:
            done = True
            assert self.constraints_satisfied == self.n_dests, "BUG IN THE LIBRARY! Not all constraints satisfied!"
            info['solved'] = True
                
        elif info['mask'].sum() == 0:
            reward = -2*self.n_nodes * self.n_dests
            done = True
            info['solved'] = False
            return vectorize_graph(self.graph), reward, done, False, info
            
        
        if done:
            info['heuristic_solution'] = self.approx_solution
            info['solution_cost'] = self.solution_cost
        
        
        return vectorize_graph(self.graph), reward, done, False, info
