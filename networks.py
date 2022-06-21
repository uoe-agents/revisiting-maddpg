from typing import List
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import utils

# TODO: PASS HIDDEN DIMS & DEPTH AS A PARAM
# TODO: Typing

class ActorNetwork(hk.Module):
    def __init__(self, n_actions, key):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.key = key

    def __call__(self, obs: jnp.ndarray) -> jnp.DeviceArray:
        net = hk.Sequential(layers=[
            hk.Linear(100), # TODO: w_init = ?
            jnn.relu,
            hk.Linear(self.n_actions),
            jnn.log_softmax,
            utils.GumbelSoftmax(temperature=0.75, key=self.key)
        ])
        return net(obs) #  TODO: Gumbel-softmax etc

class CriticNetwork(hk.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

    def __call__(self, all_obs, acts_per_agent: List) -> jnp.DeviceArray:
        net = hk.Sequential(layers=[
            hk.Linear(100), # TODO: w_init = ? 
            jnn.relu,
            hk.Linear(1),
        ])
        critic_input = jnp.concatenate((all_obs, *acts_per_agent))
        return net(critic_input)