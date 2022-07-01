from copy import deepcopy
from email import policy
from agent import Agent
from typing import List
from jax import vmap, jit
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import einops
import utils
from networks import ActorNetwork, CriticNetwork

class MADDPG:
    def __init__(self, env, critic_lr, actor_lr, gradient_clip, hidden_dim_width, gamma, gumbel_temp, rng):
        self.n_agents = env.n_agents
        self.gamma = gamma
        self.rng = rng
        obs_dims = [obs.shape[0] for obs in env.observation_space]
        act_dims = [act.n for act in env.action_space]
        self.agents = [
            Agent(
                agent_idx=ii,
                obs_dims=obs_dims,
                act_dims=act_dims,
                # TODO: Consider changing this to **config
                hidden_dim_width=hidden_dim_width,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                gradient_clip=gradient_clip,
                gumbel_temp=gumbel_temp,
                rng=rng,
            )
            for ii in range(self.n_agents)
        ]

    def acts(self, obs: List):
        actions = [self.agents[ii].act_behaviour(obs[ii], next(self.rng)) for ii in range(self.n_agents)]
        return actions

    def update(self, sample):
        batched_obs = jnp.concatenate(sample['obs'], axis=1)
        batched_nobs = jnp.concatenate(sample['nobs'], axis=1)
        
        # ********
        # TODO: This is all a bit cumbersome--could be cleaner?

        keys = jrand.split(next(self.rng), num=batched_obs.shape[0])
        target_actions = [
            vmap(self.agents[ii].act_target, in_axes=(0,0))(sample['nobs'][ii], keys)
            for ii in range(self.n_agents)
        ]

        target_actions_one_hot = [
            jnn.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        sampled_actions_one_hot = [
            jnn.one_hot(sample['acts'][ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        # ********

        for ii, agent in enumerate(self.agents):
            agent.update_critic(
                all_obs=batched_obs,
                all_nobs=batched_nobs,
                target_actions_per_agent=target_actions_one_hot,
                sampled_actions_per_agent=sampled_actions_one_hot,
                rewards=jnp.expand_dims(sample['rwds'][ii], axis=1),
                dones=jnp.expand_dims(sample['dones'][ii], axis=1),
                gamma=self.gamma,
            )

            agent.update_actor(
                all_obs=batched_obs,
                agent_obs=sample['obs'][ii],
                sampled_actions=sampled_actions_one_hot,
            )

        # TODO: Check -> Does this need to be done @ end of individual agent updates??
        for agent in self.agents:
            agent.soft_update()

        return None
