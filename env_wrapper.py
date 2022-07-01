import gym, rware, lbforaging
from make_env import make_env

def create_env(env_string):
    env = ...
    if (env_string.find("Foraging") != -1 or \
        env_string.find("rware") != -1):
        env = gym.make(env_string)
    else: # Assume MPE
        env = make_env(env_string)
    
    return env