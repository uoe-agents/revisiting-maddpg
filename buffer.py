import numpy as np
from collections import Counter
from torch import Tensor

class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size: int): # Todo fix types

        self.capacity = int(capacity)
        self.entries = 0

        self.batch_size = batch_size

        self.obs_dims = obs_dims
        self.max_obs_dim = np.max(obs_dims)
        self.n_agents = len(obs_dims)

        self.memory_obs = []
        self.memory_nobs = []
        for ii in range(self.n_agents):
            self.memory_obs.append( Tensor(self.capacity, obs_dims[ii]) )
            self.memory_nobs.append( Tensor(self.capacity, obs_dims[ii]) )
        self.memory_acts = Tensor(self.n_agents, self.capacity)
        self.memory_rwds = Tensor(self.n_agents, self.capacity)
        self.memory_dones = Tensor(self.n_agents, self.capacity)

    def store(self, obs, acts, rwds, nobs, dones):
        store_index = self.entries % self.capacity

        for ii in range(self.n_agents):
            self.memory_obs[ii][store_index] = Tensor(obs[ii])
            self.memory_nobs[ii][store_index] = Tensor(nobs[ii])
        self.memory_acts[:,store_index] = Tensor(acts)
        self.memory_rwds[:,store_index] = Tensor(rwds)
        self.memory_dones[:,store_index] = Tensor(dones)
        
        self.entries += 1

    def sample(self):
        if not self.ready(): return None

        idxs = np.random.choice(
            np.min((self.entries, self.capacity)),
            size=(self.batch_size,),
            replace=False, # TODO: different from jax version
        )

        return {
            "obs": [self.memory_obs[ii][idxs] for ii in range(self.n_agents)],
            "acts": self.memory_acts[:,idxs],
            "rwds": self.memory_rwds[:,idxs],
            "nobs": [self.memory_nobs[ii][idxs] for ii in range(self.n_agents)],
            "dones": self.memory_dones[:,idxs],
        }
    
    def ready(self):
        return (self.batch_size <= self.entries)