import gym, rware, lbforaging
from make_env import make_env
from gym.envs.registration import register

register(
    id="Foraging-8x8-2p-2f-2s-c-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-3p-3f-2s-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

def create_env(env_string):
    env = ...
    if (env_string.find("Foraging") != -1 or \
        env_string.find("rware") != -1):
        env = gym.make(env_string)
    else: # Assume MPE
        env = make_env(env_string)
    
    return env