from typing import List
from copy import copy, deepcopy
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
import einops
from functools import partial
from networks import ActorNetwork, CriticNetwork
from utils import gumbel_softmax_st

class Agent:
    def __init__(self,
        agent_idx,
        obs_dims,
        act_dims,
        hidden_dim_width,
        critic_lr,
        actor_lr,
        gradient_clip,
        gumbel_temp, # TODO: Pass as param
        soft_update_size=0.01,  # TODO: Pass as param
        # more TODO
    ):
        self.agent_idx = agent_idx
        self.soft_update_size = soft_update_size
        self.n_obs = obs_dims[self.agent_idx]
        self.n_acts = act_dims[self.agent_idx]
        self.n_agents = len(obs_dims)
        self.gradient_clip = gradient_clip
        self.gumbel_temp = gumbel_temp
        # -----------

        # ***** POLICY *****
        self.policy = ActorNetwork(self.n_obs, hidden_dim_width, self.n_acts)
        self.target_policy = ActorNetwork(self.n_obs, hidden_dim_width, self.n_acts)        
        # ***** ****** *****

        # ***** CRITIC *****
        self.critic = CriticNetwork(obs_dims, act_dims, hidden_dim_width)        
        self.target_critic = CriticNetwork(obs_dims, act_dims, hidden_dim_width)        
        # ***** ****** *****

        # OPTIMISERS
        self.optim_actor = Adam(self.policy.parameters(), lr=actor_lr, eps=0.001)
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr, eps=0.001)

    def act_behaviour(self, obs):
        policy_output = self.policy(Tensor(obs))
        gs_output = gumbel_softmax_st(policy_output, temperature=self.gumbel_temp)
        return torch.argmax(gs_output, dim=-1)

    def act_target(self, obs):
        policy_output = self.policy(Tensor(obs))
        gs_output = gumbel_softmax_st(policy_output, temperature=self.gumbel_temp)
        return torch.argmax(gs_output, dim=-1)

    def update_critic(self, all_obs, all_nobs, target_actions_per_agent, sampled_actions_per_agent, rewards, dones, gamma):
        target_actions = torch.concat(target_actions_per_agent, axis=1)
        sampled_actions = torch.concat(sampled_actions_per_agent, axis=1)
        
        Q_next_target = self.critic(torch.concat((all_nobs, target_actions), dim=1))
        target_ys = torch.as_tensor(rewards) + torch.as_tensor((1 - dones) * gamma) * Q_next_target
        behaviour_ys = self.critic(torch.concat((all_obs, sampled_actions), dim=1))
        
        loss = F.mse_loss(behaviour_ys, target_ys.detach())

        self.optim_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.optim_critic.step()

    def update_actor(self, all_obs, agent_obs, sampled_actions):

        policy_outputs = self.policy(agent_obs)
        gs_outputs = gumbel_softmax_st(policy_outputs, self.gumbel_temp)
        
        _sampled_actions = deepcopy(sampled_actions)
        _sampled_actions[self.agent_idx] = gs_outputs

        loss = - self.critic(torch.concat((all_obs, *_sampled_actions), axis=1)).mean()

        self.optim_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optim_actor.step()

    def soft_update(self):
        self.target_critic.soft_update(self.critic, self.soft_update_size)
        self.target_policy.soft_update(self.policy, self.soft_update_size)