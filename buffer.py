import numpy as np
import jax.numpy as jnp
import jax.random as jrand

class ReplayBuffer:
    def __init__(self, capacity: int | float, obs_dims, batch_size: int, key): # Todo fix types
        self.key = key

        self.capacity = int(capacity)
        self.entries = 0

        self.batch_size = batch_size

        self.obs_dims = obs_dims
        self.n_agents = len(obs_dims)

        self.memory_obs = np.zeros((self.capacity, sum(self.obs_dims)))
        self.memory_acts = np.zeros((self.capacity, self.n_agents))
        self.memory_rwds = np.zeros((self.capacity, self.n_agents))
        self.memory_nobs = np.zeros((self.capacity, sum(self.obs_dims)))
        self.memory_dones = np.zeros((self.capacity, self.n_agents), dtype=bool)

    def store(self, obs, acts, rwds, nobs, dones):
        store_index = self.entries % self.capacity

        self.memory_obs[store_index] = obs
        self.memory_acts[store_index] = acts
        self.memory_rwds[store_index] = rwds
        self.memory_nobs[store_index] = nobs
        self.memory_dones[store_index] = dones
        
        self.entries += 1

    def sample(self):
        if not self.ready(): return None

        self.key, sample_key = jrand.split(self.key)
        idxs = jrand.choice(sample_key,
            jnp.min(jnp.array((self.entries, self.capacity))),
            shape=(self.batch_size,),
            replace=False,
        )

        return {
            "obs": self.memory_obs[idxs],
            "acts": self.memory_acts[idxs],
            "rwds": self.memory_rwds[idxs],
            "nobs": self.memory_nobs[idxs],
            "dones": self.memory_dones[idxs],
        }
    
    def ready(self):
        return (self.batch_size <= self.entries)