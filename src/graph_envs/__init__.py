
import gymnasium as gym
from gymnasium.envs.registration import register

name = "graph_envs"

# register(
#     id='MultiCast-v0',
#     entry_point='envs.multicast:MultiCastEnv',
# )

register(
    id='ShortestPath-v0',
    entry_point='src.graph_envs.shortest_path:ShortestPathEnv',
)