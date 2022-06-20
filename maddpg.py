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

class MADDPG:
    def __init__(self, env, critic_lr, actor_lr, hidden_dim_width, gamma, key):
        self.n_agents = env.n_agents
        self.gamma = gamma
        self.key, *agent_keys = jrand.split(key,num=self.n_agents+1)
        self.agents = [
            Agent(
                agent_idx=ii,
                observation_space=env.observation_space,
                action_space=env.action_space,
                # TODO: Consider changing this to **config
                hidden_dim_width=hidden_dim_width,
                critic_lr=critic_lr,
                actor_lr=actor_lr,
                agent_key=agent_keys[ii],
            )
            for ii in range(self.n_agents)
        ]

    def acts(self, obs: List):
        return [jnp.argmax(self.agents[ii].act_behaviour(obs[ii])) for ii in range(self.n_agents)]

    def update(self, sample):
        # TODO: I really don't like all these concatenate operations :(
        
        all_obs = jnp.concatenate(sample['obs'], axis=1)
        all_nobs = jnp.concatenate(sample['nobs'], axis=1)
        
        target_actions = [
            vmap(agent.act_target)(obs) for agent, obs in zip(self.agents, sample['obs']) # OBS OR NOBS???? hmmm
        ] # TODO: this seems to return same one-hot for all samples in batch
        #einops.rearrange(, 'agent batch action -> batch (agent action)') 

        behaviour_actions = [
            vmap(agent.act_behaviour)(obs) for agent, obs in zip(self.agents, sample['obs'])
        ]

        sampled_actions = [
            jnn.one_hot(sample['acts'][:,ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ]

        # TODO: Better nomenclature
        critic_in_A = jnp.concatenate((
            all_nobs,
            einops.rearrange(target_actions, 'agent batch action -> batch (agent action)')
        ), axis=1)

        critic_in_B = jnp.concatenate((
            all_obs,
            einops.rearrange(sampled_actions, 'agent batch action -> batch (agent action)')
        ), axis=1)

        losses = []
        for ii, agent in enumerate(self.agents):
            sampled_actions_i = deepcopy(sampled_actions)
            sampled_actions_i[ii] = behaviour_actions[ii]
            critic_in_C_i = jnp.concatenate((
                all_obs,
                einops.rearrange(sampled_actions_i, 'agent batch action -> batch (agent action)')
            ), axis=1)

            critic_loss, policy_loss = agent.update(
                critic_in_A,
                critic_in_B,
                critic_in_C_i,
                sample['rwds'][:,ii],
                self.gamma,
            )
            print(f"Critic Loss = {critic_loss}\t; Policy Loss = {policy_loss}")

        # TODO: Check -> Does this need to be done @ end of individual agent updates??
        # for agent in self.agents:
        #     agent.soft_update()

        return None
