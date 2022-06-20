from copy import copy
import jax
from torch import eq
from jax import value_and_grad, vmap, jit, grad
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import utils
import haiku as hk
import optax
import einops
import functools

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
        tau=0.005,
        # more TODO
    ):
        self.agent_idx = agent_idx
        self.tau = tau # TODO: Not sure yet if this should be class member or fn param
        self.n_obs = observation_space[self.agent_idx].shape[0]
        self.n_acts = action_space[self.agent_idx].n
        # -----------
        self.key, *subkeys = jrand.split(agent_key, num=3)

        # ***** POLICY *****
        policy_in_size = self.n_obs
        policy_out_size = self.n_acts

        def _policy_fn(xx):
            mlp = hk.nets.MLP([policy_in_size, hidden_dim_width, policy_out_size])
            return mlp(xx)
        self.behaviour_policy = hk.without_apply_rng(hk.transform(_policy_fn))
        self.behaviour_policy_params = self.behaviour_policy.init(subkeys[0], jnp.ones((policy_in_size,)))
    
        # self.behaviour_policy = eqx.nn.MLP(
        #     in_size=policy_in_size,
        #     width_size=hidden_dim_width,
        #     out_size=policy_out_size,
        #     depth=1,
        #     activation=jnn.relu,
        #     final_activation=jnn.log_softmax, # Gumbel-Softmax takes in log-probabilities, right?? (TODO)
        #     key=subkeys[0],
        # )
        self.target_policy = copy(self.behaviour_policy) #  TODO: should this use a different rng key??
        self.target_policy_params = copy(self.behaviour_policy_params) # TODO :(
        # ***** ****** *****

        # ***** CRITIC *****
        critic_in_size = \
            sum([obs.shape[0] for obs in observation_space]) + \
            sum([act.n for act in action_space])

        def _critic_fn(xx):
            mlp = hk.nets.MLP([critic_in_size, hidden_dim_width, 1]) # Critic output is scalar value
            return mlp(xx)
        self.behaviour_critic = hk.without_apply_rng(hk.transform(_critic_fn))
        self.behaviour_critic_params = self.behaviour_critic.init(subkeys[1], jnp.ones((critic_in_size,)))
        # self.behaviour_critic = eqx.nn.MLP(
        #     in_size=critic_in_size,
        #     width_size=hidden_dim_width,
        #     out_size=1, # Critic outputs a scalar value
        #     depth=1,
        #     activation=jnn.relu,
        #     key=subkeys[1],
        # )
        self.target_critic = copy(self.behaviour_critic)
        self.target_critic_params = copy(self.behaviour_critic_params) # TODO :(
        # ***** ****** *****

        # OPTIMISERS
        self.policy_optim = optax.adam(actor_lr)
        self.policy_optim_state = self.policy_optim.init(self.behaviour_policy_params)

        self.critic_optim = optax.adam(critic_lr)
        self.critic_optim_state = self.critic_optim.init(self.behaviour_critic_params)

    def _act(self, obs, network_fn): # TODO: Temperature must be a param
        """
            Returns one-hot
        """
        self.key, act_key = jrand.split(self.key)
        actions_one_hot = utils.gumbel_softmax(network_fn(obs), act_key, temperature=0.75, st=True)
        return actions_one_hot

    def act_behaviour(self, obs):
        return self._act(obs, lambda xx: self.behaviour_policy.apply(
            self.behaviour_policy_params, xx))
    
    def act_target(self, obs):
        return self._act(obs, lambda xx: self.target_policy.apply(self.target_policy_params, xx))

    def update(self,
        critic_in_A,
        critic_in_B,
        critic_in_C,
        rewards,
        gamma,
    ):
        # TODO: Better nomenclature
        # ------- CRITIC UPDATE -------
        @value_and_grad
        def _critic_loss(
            behaviour_critic_params,
            behaviour_critic,
            target_critic_params,
            target_critic,
            critic_in_A,
            critic_in_B,
            rewards,
            gamma,
        ):
            Qs_critic = vmap(target_critic.apply, in_axes=(None,0))(target_critic_params, critic_in_A)
            target_ys = rewards + gamma * Qs_critic
            behaviour_ys = vmap(behaviour_critic.apply, in_axes=(None,0))(behaviour_critic_params, critic_in_B)
            return jnp.mean((target_ys - behaviour_ys)**2)

        critic_loss, critic_grads = _critic_loss(
            self.behaviour_critic_params,
            self.behaviour_critic,
            self.target_critic_params,
            self.target_critic,
            critic_in_A,
            critic_in_B,
            rewards,
            gamma
        )
        critic_updates, self.critic_optim_state = self.critic_optim.update(critic_grads, self.critic_optim_state)
        self.behaviour_critic_params = optax.apply_updates(self.behaviour_critic_params, critic_updates)
        # ------- ------ ------ -------

        # ------- ACTOR UPDATE -------
        @value_and_grad
        def _actor_loss(
            behaviour_policy_params,
            behaviour_policy,
            behaviour_critic_params,
            behaviour_critic,
            critic_in_C
        ):
            Qs_actor = vmap(behaviour_critic.apply, in_axes=(None,0))(behaviour_critic_params, critic_in_C)
            return -jnp.mean(Qs_actor)
    
        policy_loss, policy_grads = _actor_loss(
            self.behaviour_policy_params,
            self.behaviour_policy,
            self.behaviour_critic_params,
            self.behaviour_critic,
            critic_in_C,
        )
        policy_updates, self.policy_optim_state = self.policy_optim.update(policy_grads, self.policy_optim_state)
        self.behaviour_policy_params = optax.apply_updates(self.behaviour_policy_params, policy_updates)

        # ------- ----- ------ -------
        return critic_loss.item(), policy_loss.item()

    def soft_update(self): # TODO: Tau here or as a class member?
        # Soft updates to targets
        self.target_critic = utils.soft_update(self.target_critic, self.behaviour_critic, self.tau)
        self.target_policy = utils.soft_update(self.target_policy, self.behaviour_policy, self.tau)
