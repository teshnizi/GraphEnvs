
import gymnasium as gym
from gymnasium.envs.registration import register

name = "graph_envs"

## For packaging:

register(
    id='ShortestPath-v0',
    entry_point='graph_envs.shortest_path:ShortestPathEnv',
)

register(
    id='LongestPath-v0',
    entry_point='graph_envs.longest_path:LongestPathEnv',
)


register(
    id='SteinerTree-v0',
    entry_point='graph_envs.steiner_tree:SteinerTreeEnv',
)

register(
    id='MaxIndependentSet-v0',
    entry_point='graph_envs.max_independent_set:MaxIndependentSet',
)

register(
    id='TSP-v0',
    entry_point='graph_envs.tsp:TSPEnv',
)

register(
    id='DistributionCenter-v0',
    entry_point='graph_envs.distribution_center:DistributionCenterEnv',
)


register(
    id='MulticastRouting-v0',
    entry_point='graph_envs.multicast_routing:MulticastRoutingEnv',
)


register(
    id='DensestSubgraph-v0',
    entry_point='graph_envs.densest_subgraph:DensestSubgraphEnv',
)
