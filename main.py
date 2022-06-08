import argparse
from tqdm import tqdm
#from pettingzoo.mpe import simple_adversary_v2
import gym, rware
from buffer import ReplayBuffer
import numpy as np
from make_env import make_env
import jax
import jax.random as jrand
from agent import Agent

jax.config.update('jax_platform_name', 'cpu')

def train(config: argparse.Namespace):
    env = make_env(config.env)
    n_agents = env.n_agents
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=10e6,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size
    )

    agents = [
        Agent(
            agent_idx=ii,
            observation_space=env.observation_space,
            action_space=env.action_space,
            # TODO: Consider changing this to **config
            hidden_dim_width=config.hidden_dim_width,
            critic_lr=config.critic_lr,
            actor_lr=config.actor_lr,
        )
        for ii in range(n_agents)
    ]


    for epi_i in tqdm(range(config.n_episodes)):
        obs = env.reset()
        for _ in range(config.episode_length):
            if (config.render): env.render()
            #acts = env.action_space.sample() # TODO: acts should come from agent policies
            acts = [agents[ii].act(obs[ii]) for ii in range(n_agents)]
            nobs, rwds, dones, _ = env.step(acts)

            buffer.store(
                obs=np.concatenate(obs),
                acts=acts,
                rwds=rwds,
                nobs=np.concatenate(nobs),
                dones=dones,
            )

            obs = nobs

            if buffer.ready():
                sample = buffer.sample(key=jrand.PRNGKey(0)) # TODO: fix rng key
                print(sample)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_adversary")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_episodes", default=20, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--hidden_dim_width", default=64, type=int)
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)

    config = parser.parse_args()
    
    train(config)
