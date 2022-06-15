from copy import copy
from torch import eq
from jax import vmap
import jax.random as jrand
import jax.numpy as jnp
import jax.nn as jnn
import utils
import haiku as hk
import equinox as eqx
import optax
import einops

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
        self.tau = tau # TODO: Not sure yet if this should be class member or fn param
        self.n_acts = action_space[self.agent_idx].n
        # -----------
        self.key, *subkeys = jrand.split(agent_key, num=3)

        # ***** POLICY *****
        policy_in_size = observation_space[self.agent_idx].shape[0]
        policy_out_size = self.n_acts

        self.behaviour_policy = eqx.nn.MLP(
            in_size=policy_in_size,
            width_size=hidden_dim_width,
            out_size=policy_out_size,
            depth=1,
            activation=jnn.relu,
            final_activation=jnn.log_softmax, # Gumbel-Softmax takes in log-probabilities, right?? (TODO)
            key=subkeys[0],
        )
        self.target_policy = copy(self.behaviour_policy) #  TODO: should this use a different rng key??
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
        self.target_critic = copy(self.behaviour_critic) # TODO: as above??
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

    def _act(self, obs, network): # TODO: Temperature must be a param
        """
            Returns one-hot
        """
        self.key, act_key = jrand.split(self.key)
        actions_one_hot = utils.gumbel_softmax(network(obs), act_key, temperature=0.1, st=True)
        return actions_one_hot

    def act_behaviour(self, obs):
        return self._act(obs, self.behaviour_policy)
    
    def act_target(self, obs):
        return self._act(obs, self.target_policy)


    def update(self,
        critic_in_A,
        critic_in_B,
        critic_in_C,
        rewards,
        gamma,
    ):
        # TODO: Better nomenclature
        # ------- CRITIC UPDATE -------
        Qs = vmap(self.target_critic)(critic_in_A)
        target_ys = rewards + gamma * Qs.T[0]
        behaviour_ys = vmap(self.behaviour_critic)(critic_in_B)
        
        critic_loss, critic_grads = self._critic_loss(target_ys, behaviour_ys)
        critic_updates, self.critic_optim_state = self.critic_optim.update(critic_grads, self.critic_optim_state)
        self.behaviour_critic = eqx.apply_updates(self.behaviour_critic, critic_updates)
        # TODO: Clip grad norm?

        # ------- ------ ------ -------

        # ------- ACTOR UPDATE -------
        Qs = vmap(self.behaviour_critic)(critic_in_C)
        policy_loss, policy_grads = self._actor_loss(Qs)
        policy_updates, self.policy_optim_state = self.policy_optim.update(policy_grads, self.policy_optim_state)
        self.behaviour_policy = eqx.apply_updates(self.behaviour_policy, policy_updates)

        # ------- ----- ------ -------
        return critic_loss.item(), policy_loss.item()

    def soft_update(self): # TODO: Tau here or as a class member?
        # Soft updates to targets
        self.target_critic = utils.soft_update(self.target_critic, self.behaviour_critic, self.tau)
        self.target_policy = utils.soft_update(self.target_policy, self.behaviour_policy, self.tau)

    @eqx.filter_value_and_grad
    def _critic_loss(self, target_ys, behaviour_ys):
        return jnp.mean((target_ys - behaviour_ys)**2)
        # TODO: Policy regulariser?

    @eqx.filter_value_and_grad
    def _actor_loss(self, critic_output):
        return -jnp.mean(critic_output)
        # TODO: Policy regulariser?