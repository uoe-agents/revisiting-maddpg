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
        all_obs = einops.rearrange(sample['obs'], 'batch agent obs -> batch (agent obs)')
        all_nobs = einops.rearrange(sample['nobs'], 'batch agent nobs -> batch (agent nobs)')
        
        target_actions = einops.rearrange([
            vmap(self.agents[ii].act_target)(sample['obs'][:,ii,:]) for ii in range(self.n_agents)
        ], 'agent batch action -> batch agent action')
        # TODO: this seems to return same one-hot for all samples in batch ??

        sampled_actions = einops.rearrange([
            jnn.one_hot(sample['acts'][:,ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ], 'agent batch action -> batch agent action')

        critic_loss = 0; actor_loss = 0
        for ii, agent in enumerate(self.agents):
            critic_loss += agent.update_critic(
                all_obs=all_obs,
                all_nobs=all_nobs,
                target_actions=target_actions,
                sampled_actions=sampled_actions,
                rewards=sample['rwds'][:,ii],
                gamma=self.gamma,
            ).item()

            actor_loss += agent.update_actor(
                all_obs=all_obs,
                agent_obs=sample['obs'][:,ii,:],
                sampled_actions=sampled_actions,
            ).item()
        
        #print(f"Critic Loss = {critic_loss}; Actor Loss = {actor_loss}")

        # TODO: Check -> Does this need to be done @ end of individual agent updates??
        for agent in self.agents:
            agent.soft_update()

        return None
