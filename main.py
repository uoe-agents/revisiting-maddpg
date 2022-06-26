import argparse
from tqdm import tqdm
from buffer import ReplayBuffer
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import haiku as hk
from agent import Agent
from env_wrapper import create_env
from maddpg import MADDPG
import wandb

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


def train(config: argparse.Namespace, rng):
    env = create_env(config.env)
    #n_agents = env.n_agents
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=10e6,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size,
        rng=rng,
    )

    maddpg = MADDPG(
        env=env,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        hidden_dim_width=config.hidden_dim_width,
        gamma=config.gamma,
        rng=rng,
    )

    with tqdm(range(config.n_episodes),
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}") as pbar:
        for epi_i in pbar:
            episode_return = play_episode(
                env,
                maddpg,
                buffer,
                max_timesteps=config.episode_length,
                steps_per_update=config.steps_per_update,
                train=config.training_on,
                render=False,
            )
            wandb.log({"Ep. Return (Train)": episode_return})
            pbar.set_postfix(episode_return=f"{np.round(episode_return, 2)}", refresh=True)

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
                wandb.log({"Ep. Return (Eval)": episode_return})

    env.close()

if __name__ == "__main__":
    wandb.init(project="maddpg-jax-first-tests", entity="callumtilbury")#, mode="disabled")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_adversary")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--hidden_dim_width", default=256, type=int)
    parser.add_argument("--critic_lr", default=1e-2, type=float)
    parser.add_argument("--actor_lr", default=1e-2, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--training_on", default=True, type=bool)

    config = parser.parse_args()
    
    wandb.config = config

    #base_key = jrand.PRNGKey(config.seed)
    rng = hk.PRNGSequence(config.seed)
    train(config, rng)
