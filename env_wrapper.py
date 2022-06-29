import gym, rware, lbforaging, matrixgames
from make_env import make_env

def create_env(env_string):
    env = ...
    if (env_string.find("Foraging") != -1 or \
        env_string.find("rware") != -1):
        env = gym.make(env_string)
    elif (env_string.find("matrix") != -1):
        env = gym.make("climbing-nostate-v0")
    else: # Assume MPE
        env = make_env(env_string)
    
    return env