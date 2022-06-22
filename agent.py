from typing import List
from copy import copy, deepcopy
import jax
from torch import eq
from jax import value_and_grad, vmap, jit, grad
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import utils
from utils import _hk_tt
import haiku as hk
import optax
import einops
from functools import partial
from networks import ActorNetwork, CriticNetwork

class Agent:
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
        self.tau = tau # TODO: Not sure yet if this should be class member or fn param
        self.n_obs = observation_space[self.agent_idx].shape[0]
        self.n_acts = action_space[self.agent_idx].n
        # -----------
        self.key, *subkeys = jrand.split(agent_key, num=4)

        # ***** POLICY *****
        policy_in_size = self.n_obs
        policy_out_size = self.n_acts

        self.policy = _hk_tt( lambda xx: ActorNetwork(self.n_acts, subkeys[0])(xx) )
        self.behaviour_policy_params = self.target_policy_params = \
            self.policy.init(subkeys[1], jnp.ones((policy_in_size,)))
        # ***** ****** *****

        # ***** CRITIC *****
        critic_in_size = \
            sum([obs.shape[0] for obs in observation_space]) + \
            sum([act.n for act in action_space])
        all_obs_size = sum([obs.shape[0] for obs in observation_space])
        act_size = action_space[0].n # TODO: For now, assuming that all agents have same size space --> I think this will be okay, with padding etc.

        self.critic = _hk_tt (lambda obs, acts : CriticNetwork()(obs, acts))
        self.behaviour_critic_params = self.target_critic_params = \
            self.critic.init(
                subkeys[2],
                jnp.ones((all_obs_size,)),
                jnp.ones((len(observation_space),act_size))
            )
        # ***** ****** *****

        # OPTIMISERS
        self.policy_optim = optax.adam(actor_lr)
        self.policy_optim_state = self.policy_optim.init(self.behaviour_policy_params)

        self.critic_optim = optax.adam(critic_lr)
        self.critic_optim_state = self.critic_optim.init(self.behaviour_critic_params)

    @partial(jit, static_argnums=(0,))
    def act_behaviour(self, obs):
        return self.policy.apply(self.behaviour_policy_params, obs)

    @partial(jit, static_argnums=(0,))
    def act_target(self, obs):
        return self.policy.apply(self.target_policy_params, obs)

    #@partial(jit, static_argnums=(0,))
    def update_critic(self, all_obs, all_nobs, target_actions, sampled_actions, rewards, gamma):

        @value_and_grad
        def _critic_loss_fn(
            behaviour_critic_params,
            target_critic_params,
            critic_network,
            all_obs,
            all_nobs,
            target_actions,
            sampled_actions,
            rewards,
            gamma,
        ):
            Q_vals = vmap(critic_network.apply, in_axes=(None,0,0))(target_critic_params, all_nobs, target_actions)
            target_ys = rewards + gamma * Q_vals
            behaviour_ys = vmap(critic_network.apply, in_axes=(None,0,0))(behaviour_critic_params, all_obs, sampled_actions)
            return jnp.mean((target_ys - behaviour_ys)**2)
        
        critic_loss, critic_grads = _critic_loss_fn(
            self.behaviour_critic_params,
            self.target_critic_params,
            self.critic,
            all_obs,
            all_nobs,
            target_actions,
            sampled_actions,
            rewards,
            gamma,
        )
        
        critic_updates, self.critic_optim_state = self.critic_optim.update(critic_grads, self.critic_optim_state)
        self.behaviour_critic_params = optax.apply_updates(self.behaviour_critic_params, critic_updates)

        return critic_loss

    #@partial(jit, static_argnums=(0,))
    def update_actor(self, all_obs, agent_obs, sampled_actions):
        
        @value_and_grad
        def _actor_loss_fn(
            behaviour_policy_params,
            policy_network,
            behaviour_critic_params,
            critic_network,
            all_obs,
            agent_obs,
            sampled_actions,
        ):
            Q_vals = vmap(critic_network.apply, in_axes=(None,0,0))(behaviour_critic_params, all_obs,
                sampled_actions.at[:,self.agent_idx,:].set(
                    vmap(policy_network.apply, in_axes=(None,0))(behaviour_policy_params, agent_obs))
            )
            return -jnp.mean(Q_vals)
    
        actor_loss, actor_grads = _actor_loss_fn(
            self.behaviour_policy_params,
            self.policy,
            self.behaviour_critic_params,
            self.critic,
            all_obs,
            agent_obs,
            sampled_actions,
        )

        actor_updates, self.policy_optim_state = self.policy_optim.update(actor_grads, self.policy_optim_state)
        self.behaviour_policy_params = optax.apply_updates(self.behaviour_policy_params, actor_updates)

        return actor_loss

    def soft_update(self): # TODO: Tau here or as a class member?
        # Soft updates to targets
        self.target_policy_params = optax.incremental_update(
            self.behaviour_policy_params,
            self.target_policy_params,
            step_size=self.tau,
        )
        self.target_critic_params = optax.incremental_update(
            self.behaviour_critic_params,
            self.target_critic_params,
            step_size=self.tau,
        )