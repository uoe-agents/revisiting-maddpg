import numpy as np
import torch
from agent import Agent
from typing import List
import torch.nn.functional as F
from utils import RunningMeanStd

from gradient_estimators import GradientEstimator

class MADDPG:
    def __init__(
        self,
        env,
        critic_lr : float,
        actor_lr : float,
        gradient_clip : float,
        hidden_dim_width : int,
        gamma : float,
        gumbel_temp : float,
        policy_regulariser : float,
        gradient_estimator : GradientEstimator,
        standardise_rewards : bool,
    ):
        self.n_agents = env.n_agents
        self.gamma = gamma
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
                policy_regulariser=policy_regulariser,
                gradient_estimator=gradient_estimator,
            )
            for ii in range(self.n_agents)
        ]

        self.return_std = RunningMeanStd(shape=(self.n_agents,)) if standardise_rewards else None

    def acts(self, obs: List):
        actions = [self.agents[ii].act_behaviour(obs[ii]) for ii in range(self.n_agents)]
        return actions

    def update(self, sample):
        # sample['obs'] : agent batch obs
        batched_obs = torch.concat(sample['obs'], axis=1)
        batched_nobs = torch.concat(sample['nobs'], axis=1)

        # ********
        # TODO: This is all a bit cumbersome--could be cleaner?

        target_actions = [
            self.agents[ii].act_target(sample['nobs'][ii])
            for ii in range(self.n_agents)
        ]

        target_actions_one_hot = [
            F.one_hot(target_actions[ii], num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        sampled_actions_one_hot = [
            F.one_hot(sample['acts'][ii].to(torch.int64), num_classes=self.agents[ii].n_acts)
            for ii in range(self.n_agents)
        ] # agent batch actions

        # ********
        # Standardise rewards if requested
        rewards = sample['rwds']
        if self.return_std is not None:
            self.return_std.update(rewards)
            rewards = ((rewards.T - self.return_std.mean) / torch.sqrt(self.return_std.var)).T
        # ********

        for ii, agent in enumerate(self.agents):
            agent.update_critic(
                all_obs=batched_obs,
                all_nobs=batched_nobs,
                target_actions_per_agent=target_actions_one_hot,
                sampled_actions_per_agent=sampled_actions_one_hot,
                rewards=rewards[ii].unsqueeze(dim=1),
                dones=sample['dones'][ii].unsqueeze(dim=1),
                gamma=self.gamma,
            )

            agent.update_actor(
                all_obs=batched_obs,
                agent_obs=sample['obs'][ii],
                sampled_actions=sampled_actions_one_hot,
            )

        for agent in self.agents:
            agent.soft_update()

        return None
