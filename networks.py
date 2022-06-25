from typing import List
import haiku as hk
import jax.random as jrand
import jax.nn as jnn
import jax.numpy as jnp
import utils

# TODO: PASS HIDDEN DIMS & DEPTH AS A PARAM
# TODO: Typing

class ActorNetwork(hk.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim_width):
        super(ActorNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim_width = hidden_dim_width

    def __call__(self, obs: jnp.ndarray) -> jnp.DeviceArray:
        net = hk.Sequential(layers=[
            hk.Linear(self.hidden_dim_width), # TODO: w_init = ?
            jnn.relu,
            hk.Linear(self.hidden_dim_width),
            jnn.relu,
            hk.Linear(self.n_actions),
        ])
        logits = net(obs[:self.obs_dim])
        return  utils.gumbel_softmax_st(
            logits,
            key = hk.next_rng_key(),
            tau = 0.75,
        )

class CriticNetwork(hk.Module):
    def __init__(self, obs_dims, hidden_dim_width):
        super(CriticNetwork, self).__init__()
        max_obs_dim = jnp.max(obs_dims)
        self.obs_mask = jnp.concatenate([
            jnp.arange(ii*max_obs_dim , ii*max_obs_dim + obs_dims[ii]) for ii in range(len(obs_dims))
        ])
        self.hidden_dim_width = hidden_dim_width

    def __call__(self, all_obs, acts_per_agent: List) -> jnp.DeviceArray:
        net = hk.Sequential(layers=[
            hk.Linear(self.hidden_dim_width), # TODO: w_init = ? 
            jnn.relu,
            hk.Linear(self.hidden_dim_width),
            jnn.relu,
            hk.Linear(1),
        ])
        critic_input = jnp.concatenate((all_obs[self.obs_mask], *acts_per_agent))
        return net(critic_input)