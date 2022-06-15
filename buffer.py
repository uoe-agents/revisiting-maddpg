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

        self.memory_obs = {}
        self.memory_nobs = {}
        for ii in range(self.n_agents):
            self.memory_obs[ii] = np.zeros((self.capacity, obs_dims[ii]))
            self.memory_nobs[ii] = np.zeros((self.capacity, obs_dims[ii]))

        self.memory_acts = np.zeros((self.capacity, self.n_agents))
        self.memory_rwds = np.zeros((self.capacity, self.n_agents))
        self.memory_dones = np.zeros((self.capacity, self.n_agents), dtype=bool)

    def store(self, obs, acts, rwds, nobs, dones):
        store_index = self.entries % self.capacity

        for ii in range(self.n_agents):
            self.memory_obs[ii][store_index] = obs[ii]
            self.memory_nobs[ii][store_index] = nobs[ii]
        self.memory_acts[store_index] = acts
        self.memory_rwds[store_index] = rwds
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

        # return_array = []
        # for ii in idxs:
        #     return_array.append(
        #         {
        #             "obs": [self.memory_obs[agent][ii] for agent in range(self.n_agents)],
        #             "acts": self.memory_acts[ii],
        #             "rwds": self.memory_rwds[ii],
        #             "nobs": [self.memory_nobs[agent][ii] for agent in range(self.n_agents)],
        #             "dones": self.memory_dones[ii],
        #         }
        #     )
        # return return_array
        return {
            "obs": [self.memory_obs[ii][idxs] for ii in range(self.n_agents)],
            "acts": self.memory_acts[idxs],
            "rwds": self.memory_rwds[idxs],
            "nobs": [self.memory_nobs[ii][idxs] for ii in range(self.n_agents)],
            "dones": self.memory_dones[idxs],
        }
    
    def ready(self):
        return (self.batch_size <= self.entries)