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

def play_episode(
    env,
    maddpg : MADDPG,
    buffer : ReplayBuffer,
    max_timesteps,
    steps_per_update,
    train=True,
    render=False,
):
    obs = env.reset()
    dones = [False] * maddpg.n_agents

    steps = 0
    episode_return = 0

    while not any(dones):
        if (render): env.render()

        acts = maddpg.acts(obs)
        nobs, rwds, dones, _ = env.step(acts)

        if train:
            buffer.store(
                obs=obs,
                acts=acts,
                rwds=rwds,
                nobs=nobs,
                dones=dones,
            )

            if buffer.ready() and (steps % steps_per_update == 0): # TODO: improve training interval setup
                sample = buffer.sample()
                maddpg.update(sample)

        steps += 1

        episode_return += sum(rwds)

        if (steps > max_timesteps):
            break

        obs = nobs

    return episode_return


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

    # with tqdm(range(config.n_episodes)) as pbar:
    for epi_i in tqdm(range(config.n_episodes)):
        _ = play_episode(
            env,
            maddpg,
            buffer,
            max_timesteps=config.episode_length,
            steps_per_update=config.steps_per_update,
            train=True,
            render=False,
        )

        if (config.eval_freq != 0 and epi_i % config.eval_freq == 0):
            episode_return = play_episode(
                env,
                maddpg,
                buffer,
                max_timesteps=config.episode_length,
                steps_per_update=None,
                train=False,
                render=config.render,
            )
            print(f"Episode Return = {episode_return}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_adversary")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--hidden_dim_width", default=64, type=int)
    parser.add_argument("--critic_lr", default=1e-4, type=float)
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--eval_freq", default=100, type=int)

    config = parser.parse_args()
    
    base_key = jrand.PRNGKey(config.seed)
    train(config, base_key)
