from networks import ActorNetwork, CriticNetwork
import jax.random as jrand
import utils

class Agent:
    def __init__(self, actor_dim, critic_dim, n_agents, n_actions, agent_idx, key,
        fc_dims=[64,64], gamma=0.95, alpha_actor=0.01, alpha_critic=0.01, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f"agent_{agent_idx}"

        ba_key, bc_key, ta_key, tc_key, self.act_key = jrand.split(key, 5)

        self.behaviour_actor = ActorNetwork(
            obs_dim=actor_dim,
            n_actions=n_actions,
            fc_dims=fc_dims,
            lr=alpha_actor,
            key=ba_key,
        )

        self.behaviour_critic = CriticNetwork(
            obs_dims=critic_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            fc_dims=fc_dims,
            lr=alpha_critic,
            key=bc_key,
        )

        self.target_actor = ActorNetwork(
            obs_dim=actor_dim,
            n_actions=n_actions,
            fc_dims=fc_dims,
            lr=alpha_actor,
            key=ta_key, # Should the targets' weights be initialised to behaviour networks?
        )

        self.target_critic = CriticNetwork(
            obs_dims=critic_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            fc_dims=fc_dims,
            lr=alpha_critic,
            key=tc_key, # TODO: see above
        )

    def act(self, obs):
        
        # actions = self.behaviour_actor(obs)
        # noise = jrand.normal(self.act_key, shape=actions.shape)
        # return actions + noise
