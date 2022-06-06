import numpy as np
import jax.numpy as jnp
import jax.random as jrand

class ReplayBuffer:
    def __init__(self, capacity: int | float, obs_dims): # Todo fix types
        self.capacity = int(capacity)
        self.entries = 0

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

    def sample(self, batch_size, key):
        if (batch_size > self.entries): # Not ready yet!
            return None

        idxs = jrand.choice(key,
            jnp.min(jnp.array((self.entries, self.capacity))),
            shape=(batch_size,),
            replace=False,
        )

        return (
            self.memory_obs[idxs],
            self.memory_acts[idxs],
            self.memory_rwds[idxs],
            self.memory_nobs[idxs],
            self.memory_dones[idxs],
        )
    
    def ready(self, batch_size):
        return (batch_size <= self.entries)