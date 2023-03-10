
import gymnasium as gym
from gymnasium.envs.registration import register

name = "graph_envs"

## For local testing and running:

# register(
#     id='ShortestPath-v0',
#     entry_point='src.graph_envs.shortest_path:ShortestPathEnv',
# )

# register(
#     id='SteinerTree-v0',
#     entry_point='src.graph_envs.steiner_tree:SteinerTreeEnv',
# )

# register(
#     id='MaxIndependentSet-v0',
#     entry_point='src.graph_envs.max_independent_set:MaxIndependentSet',
# )

# register(
#     id='TSP-v0',
#     entry_point='src.graph_envs.tsp:TSPEnv',
# )


## For packaging:

register(
    id='ShortestPath-v0',
    entry_point='graph_envs.shortest_path:ShortestPathEnv',
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