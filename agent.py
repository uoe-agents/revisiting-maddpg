from copy import deepcopy
from torch import eq
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import utils
import haiku as hk
import equinox as eqx
import optax


class Agent:
    # def __init__(self, actor_dim, critic_dim, n_agents, n_actions, agent_idx, key,
        # fc_dims=[64,64], gamma=0.95, alpha_actor=0.01, alpha_critic=0.01, tau=0.01):
    def __init__(self,
        agent_idx,
        observation_space,
        action_space,
        hidden_dim_width,
        critic_lr,
        actor_lr,
        agent_key,
        tau=0.01,
        # more TODO
    ):
        self.agent_idx = agent_idx
        # self.critic_lr = critic_lr,
        # self.actor_lr = actor_lr,
        # -----------
        self.key = agent_key
        self.key, *subkeys = jrand.split(self.key, num=3)

        # ***** POLICY *****
        policy_in_size = observation_space[self.agent_idx].shape[0]
        policy_out_size = action_space[self.agent_idx].n

        self.behaviour_policy = eqx.nn.MLP(
            in_size=policy_in_size,
            width_size=hidden_dim_width,
            out_size=policy_out_size,
            depth=1,
            activation=jnn.relu,
            final_activation=jnn.log_softmax, # Gumbel-Softmax takes in log-probabilities, right??
            key=subkeys[0],
        )
        self.target_policy = deepcopy(self.behaviour_policy) #  TODO: should this use a different rng key??
        # ***** ****** *****

        # ***** CRITIC *****
        critic_in_size = \
            sum([obs.shape[0] for obs in observation_space]) + \
            sum([act.n for act in action_space])

        self.behaviour_critic = eqx.nn.MLP(
            in_size=critic_in_size,
            width_size=hidden_dim_width,
            out_size=1, # Critic outputs a scalar value
            depth=1,
            activation=jnn.relu,
            key=subkeys[1],
        )
        self.target_critic = deepcopy(self.behaviour_critic) # TODO: as above??
        # ***** ****** *****

        # OPTIMISERS
        self.policy_optim = optax.adam(actor_lr)
        self.policy_optim_state = self.policy_optim.init(
            eqx.filter(self.policy_optim, eqx.is_inexact_array)
        )

        self.critic_optim = optax.adam(critic_lr)
        self.critic_optim_state = self.critic_optim.init(
            eqx.filter(self.critic_optim, eqx.is_inexact_array)
        )

    def act(self, obs):
        self.key, act_key = jrand.split(self.key)
        actions_one_hot = utils.gumbel_softmax(self.behaviour_policy(obs), act_key, temperature=0.8, st=True)
        return jnp.argmax(actions_one_hot)

    def update(self, sample):
        print(f"**** Updating agent {self.agent_idx}!!! ****")
        print(sample)
        input()
        return None

        loss_fn = eqx.filter_value_and_grad(...)#(batch_loss_fn????)
        loss, grads = loss_fn(self.behaviour_policy)#, weight, init
        updates, self.policy_optim_state = self.policy_optim.update(grads, self.policy_optim_state)
        self.behaviour_policy = eqx.apply_updates(self.behaviour_policy, updates)
        key = jrand.split(key, 1)[0]

        print(loss)
        return None
        

    # def hard_update(self, source):
    #     for target_param, source_param in zip(self.parameters(), source.parameters()):
    #         target_param.data.copy_(source_param.data)

    # def soft_update(self, source, t):
    #     for target_param, source_param in zip(self.parameters(), source.parameters()):
    #         target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)