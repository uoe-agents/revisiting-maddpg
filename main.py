#from pettingzoo.mpe import simple_adversary_v2
import gym, rware
from buffer import ReplayBuffer
import numpy as np
from make_env import make_env

env = make_env("simple_adversary")
# env = gym.make("rware-small-4ag-v1")
N = len(env.observation_space)
obs = env.reset()
observation_dims = np.array([os.shape[0] for os in env.observation_space])

buffer = ReplayBuffer(10e6, observation_dims)

for ii in range(100):
    acts = env.action_space.sample()
    nobs, rwds, dones, _ = env.step(acts)
    
    buffer.store(
        obs=np.concatenate(obs),
        acts=acts,
        rwds=rwds,
        nobs=np.concatenate(nobs),
        dones=dones,
    )

    obs = nobs

print(buffer.sample(5))
