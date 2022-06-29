from typing import List
from copy import copy, deepcopy
import jax
from torch import eq
from jax import value_and_grad, vmap, jit, grad
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import haiku as hk
import optax
import einops
from functools import partial
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self,
        agent_idx,
        obs_dims,
        act_dims,
        hidden_dim_width,
        critic_lr,
        actor_lr,
        rng,
        gumbel_temp=0.75, # TODO: Pass as param
        soft_update_size=0.01,  # TODO: Pass as param
        # more TODO
    ):
        self.agent_idx = agent_idx
        self.soft_update_size = soft_update_size
        self.n_obs = obs_dims[self.agent_idx]
        self.n_acts = act_dims[self.agent_idx]
        self.n_agents = len(obs_dims)
        # -----------
        self.rng = rng

        # ***** POLICY *****
        self.policy = hk.transform(
            lambda xx: ActorNetwork(
                self.n_obs,
                self.n_acts,
                hidden_dim_width,
                gumbel_temp,
            )(xx)
        )
        self.behaviour_policy_params = self.target_policy_params = \
            self.policy.init(next(self.rng), jnp.ones((self.n_obs,)))
        # ***** ****** *****

        # ***** CRITIC *****
        self.critic = hk.without_apply_rng(hk.transform(lambda obs, acts : CriticNetwork(obs_dims, hidden_dim_width)(obs, acts)))
        self.behaviour_critic_params = self.target_critic_params = \
            self.critic.init(
                next(self.rng),
                jnp.ones((sum(obs_dims),)),
                jnp.ones((sum(act_dims),)),
            )
        self.batched_critic_apply = vmap(self.critic.apply, in_axes=(None,0,0))
        # ***** ****** *****

        # OPTIMISERS
        self.policy_optim = optax.adam(actor_lr)
        self.policy_optim_state = self.policy_optim.init(self.behaviour_policy_params)

        self.critic_optim = optax.adam(critic_lr)
        self.critic_optim_state = self.critic_optim.init(self.behaviour_critic_params)

    @partial(jit, static_argnums=(0,))
    def act_behaviour(self, obs, key):
        return self.policy.apply(self.behaviour_policy_params, key, obs)

    @partial(jit, static_argnums=(0,))
    def act_target(self, obs, key):
        return self.policy.apply(self.target_policy_params, key, obs)

    def update_critic(self, all_obs, all_nobs, target_actions_per_agent, sampled_actions_per_agent, rewards, dones, gamma):

        @jit
        @value_and_grad
        def _critic_loss_fn(
            behaviour_critic_params,
            target_critic_params,
            all_obs,
            all_nobs,
            target_actions,
            sampled_actions,
            rewards,
            dones,
            gamma,
        ):
            # Q_vals = vmap(self.critic.apply, in_axes=(None,0,0))(target_critic_params, all_nobs, target_actions)
            Q_vals = self.batched_critic_apply(target_critic_params, all_nobs, target_actions)
            target_ys = rewards + (1 - dones) * gamma * Q_vals
            
            # behaviour_ys = vmap(self.critic.apply, in_axes=(None,0,0))(behaviour_critic_params, all_obs, sampled_actions)
            behaviour_ys = self.batched_critic_apply(behaviour_critic_params, all_obs, sampled_actions)

            return jnp.mean((jax.lax.stop_gradient(target_ys) - behaviour_ys)**2)

        target_actions = jnp.concatenate(target_actions_per_agent, axis=1)
        sampled_actions = jnp.concatenate(sampled_actions_per_agent, axis=1)
        critic_loss, critic_grads = _critic_loss_fn(
            self.behaviour_critic_params,
            self.target_critic_params,
            all_obs,
            all_nobs,
            target_actions,
            sampled_actions,
            rewards,
            dones,
            gamma,
        )
        
        critic_updates, self.critic_optim_state = self.critic_optim.update(critic_grads, self.critic_optim_state)
        self.behaviour_critic_params = optax.apply_updates(self.behaviour_critic_params, critic_updates)

        return critic_loss

    def update_actor(self, all_obs, agent_obs, sampled_actions):
        
        @value_and_grad
        def _actor_loss_fn(
            behaviour_policy_params,
            behaviour_critic_params,
            all_obs,
            agent_obs,
            sampled_actions_per_agent,
        ):
            _sampled_actions_per_agent = deepcopy(sampled_actions_per_agent) # TODO - can we avoid this?? :(
            keys = jrand.split(next(self.rng), num=agent_obs.shape[0])
            # _sampled_actions_per_agent[self.agent_idx] = vmap(self.policy.apply, in_axes=(None,None,0))(behaviour_policy_params, next(self.rng), agent_obs)
            _sampled_actions_per_agent[self.agent_idx] = vmap(self.policy.apply, in_axes=(None,0,0))(behaviour_policy_params, keys, agent_obs)
            sampled_actions = jnp.concatenate(_sampled_actions_per_agent, axis=1)

            # Q_vals = vmap(self.critic.apply, in_axes=(None,0,0))(behaviour_critic_params, all_obs, sampled_actions)
            Q_vals = self.batched_critic_apply(behaviour_critic_params, all_obs, sampled_actions)
            
            return -jnp.mean(Q_vals)
    
        actor_loss, actor_grads = _actor_loss_fn(
            self.behaviour_policy_params,
            self.behaviour_critic_params,
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
            step_size=self.soft_update_size,
        )
        self.target_critic_params = optax.incremental_update(
            self.behaviour_critic_params,
            self.target_critic_params,
            step_size=self.soft_update_size,
        )