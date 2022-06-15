import argparse
from tqdm import tqdm
#from pettingzoo.mpe import simple_adversary_v2
import gym, rware
from buffer import ReplayBuffer
import numpy as np
from make_env import make_env
import jax
import jax.numpy as jnp
import jax.random as jrand
from agent import Agent
from maddpg import MADDPG
jax.config.update('jax_platform_name', 'cpu')

def train(config: argparse.Namespace, key):
    env = make_env(config.env)
    #n_agents = env.n_agents
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=10e6,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size,
        key=key,
    )

    maddpg = MADDPG(
        env=env,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        hidden_dim_width=config.hidden_dim_width,
        gamma=config.gamma,
        key=key,
    )

    for epi_i in tqdm(range(config.n_episodes)):
        render_on = (epi_i % 10)
        obs = env.reset()
        cum_rwd = 0
        for tt in range(config.episode_length):
            if (config.render and render_on): env.render()
            acts = maddpg.acts(obs)
            nobs, rwds, dones, _ = env.step(acts)

            buffer.store(
                obs=obs,
                acts=acts,
                rwds=rwds,
                nobs=nobs,
                dones=dones,
            )

            cum_rwd += sum(rwds)
            obs = nobs

            if buffer.ready() and (tt % config.batch_size == 0): # TODO: improve training interval setup
                sample = buffer.sample()
                maddpg.update(sample)

            if all(dones): break
        print(f"Cumulative reward: {cum_rwd}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_adversary")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_episodes", default=50, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--hidden_dim_width", default=64, type=int)
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)

    config = parser.parse_args()
    
    base_key = jrand.PRNGKey(config.seed)
    train(config, base_key)
