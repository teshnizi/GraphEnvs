import gymnasium as gym
import torch 
import torch_geometric as pyg 
import numpy as np 

import networkx as nx 
import random
from typing import Tuple

# from MILP import flow 

def sample_graph(N, E):
    G = nx.random_tree(N)
    while G.number_of_edges() < E:
        x, y = random.randint(0, N-1), random.randint(0, N-1)
        if x == y:
            continue
        if (x, y) in G.edges:
            continue
        G.add_edge(x, y)
        
    assert nx.is_connected(G)
    return G

class MultiCastEnv(gym.Env):
    def __init__(self, n_nodes, n_edges, n_targets, n_msgs, is_eval_env=False) -> None:
        super(MultiCastEnv, self).__init__()
        
        self.HAS_NOTHING = np.array([1.0])
        self.HAS_MSG = np.array([2.0])
        self.IS_TARGET = np.array([3.0])
        self.inf_delay = np.array([-1.0])
        
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_targets = n_targets
        self.n_msgs = n_msgs
        
        self.action_space = gym.spaces.MultiDiscrete([n_nodes, n_nodes])
        self.observation_space = gym.spaces.Box(low=-100., high=100., shape=(n_msgs+2, n_nodes, n_nodes))        
        self.is_eval_env = is_eval_env
        
    def reset(self, seed=None, options={}) -> np.array:
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        
        G = sample_graph(self.n_nodes, self.n_edges)
        
        bandwidth = np.random.randint(1,2, size=(self.n_nodes, self.n_nodes))
        delay = np.random.randint(2,5, size=(self.n_nodes, self.n_nodes))/1.0
        
        for u, v, d in G.edges(data=True):
            d['delay'] = delay[u, v]
            d['bandwidth'] = bandwidth[u, v]
        
        selected_nodes = np.random.choice(self.n_nodes, self.n_msgs*(1 + self.n_targets), replace=False)
        
        ''' Single Graph case for debug '''
        # G = G_global.copy()
        # selected_nodes = selected_nodes_global.copy()
        
        # asp = nx.floyd_warshall_numpy(G, delay='delay')
        adj = np.array(nx.adjacency_matrix(G, dtype=float, weight='delay').todense(), dtype=np.float32)
        adj[adj==0] = self.inf_delay
        np.fill_diagonal(adj, 0)
        
        bw = np.array(nx.adjacency_matrix(G, dtype=float, weight='bandwidth').todense(), dtype=np.float32)
        bw[bw==0] = self.inf_delay
        np.fill_diagonal(bw, 0)
        
        state = np.zeros((self.n_msgs, self.n_nodes, self.n_nodes)) + self.HAS_NOTHING
        
        msg_info = {
            i: {
                'sources': [selected_nodes[i*(1 + self.n_targets)]],
                'destinations': selected_nodes[i*(1 + self.n_targets) + 1: i*(1 + self.n_targets) + 1 + self.n_targets],
                'size': 1
            } for i in range(self.n_msgs)}

        
        for msg in range(self.n_msgs):
            pad = msg * (1 + self.n_targets)
            state[msg, selected_nodes[pad + 0],:] = self.HAS_MSG
            state[msg, selected_nodes[pad + 1: pad + self.n_targets + 1],:] = self.IS_TARGET

        
        # self.x = np.stack([np.zeros_like(adj) + self.HAS_NOTHING, self.x], axis=1, dtype=np.float32)
        self.x = np.concatenate((state, np.expand_dims(adj, 0) , np.expand_dims(bw, 0)), axis=0, dtype=np.float32)
        
        
        self.edge_index = pyg.utils.from_networkx(G).edge_index
        
        info = {'mask': self.get_mask()}
        return self.x, info
    
    def get_mask(self) -> np.array:
        
        valid_edges = np.logical_and(
            np.isclose(self.x[0:self.n_msgs, self.edge_index[0], 0], self.HAS_MSG),
            np.isclose(self.x[0:self.n_msgs, self.edge_index[1], 0], self.HAS_MSG) == False
            ) # shape: (2*n_edges,)
        
        mask = np.zeros((self.n_nodes, self.n_msgs * self.n_nodes), dtype=bool)
        bw_mask = (self.x[-1,:,:] > 0)
        
        for msg in range(self.n_msgs):
            pad = msg*self.n_nodes
            mask[self.edge_index[0, valid_edges[msg]], pad+self.edge_index[1, valid_edges[msg]]] = True
            mask[:, pad:pad+self.n_nodes] = np.logical_and(mask[:, pad:pad+self.n_nodes], bw_mask)
        
        return mask
    
    def step(self, action: Tuple[int, int, int]) -> Tuple[np.array, bool, dict]:
        
        u, v, msg = action
        
        assert (u < self.n_nodes) and (v < self.n_nodes), f"{u, v} is not a valid edge!"
        assert self.get_mask()[u, msg*self.n_nodes+v] == True, f"Mask of {u, v} is False!"
        assert not np.isclose(self.x[-2, u, v], self.inf_delay), f"{u, v} is not an edge!"
        assert np.isclose(self.x[msg, u, 0], self.HAS_MSG), f"{u} is not a source!"
        assert np.isclose(self.x[msg, v, 0], self.HAS_MSG) == False, f"{v} is a destination!"
        assert not np.isclose(self.x[-1, u, v], 0), f"{u, v}'s bandwidth is not enough!"
        
        done = False
        reward = -self.x[-2, u, v]
        
        if self.is_eval_env:
            self.solution += self.x[-2, u, v]
        
        if np.isclose(self.x[msg, v, 0], self.IS_TARGET):
            reward += 1
            
        self.x[msg, v, :] = self.HAS_MSG
        self.x[-1, u, v] -= 1
        
        
        if not np.isclose(self.x[0:self.n_msgs, :, 0], self.IS_TARGET).any():
            done = True
        elif (self.get_mask().sum() == 0):
            done = True
            reward = -10
            
        info = {'mask': self.get_mask()}
        if self.is_eval_env:
            info['opt'] = self.opt
            info['sol'] = self.solution

            
        return self.x, reward, done, False, info
        