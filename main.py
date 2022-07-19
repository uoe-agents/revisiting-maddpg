import argparse
import torch
from tqdm import tqdm; BAR_FORMAT = "{l_bar}{bar:50}{r_bar}{bar:-10b}"
from buffer import ReplayBuffer
import numpy as np
import scipy.stats as st
from env_wrapper import create_env
from maddpg import MADDPG
import wandb
from datetime import date
import gradient_estimators

def play_episode(
    env,
    buffer : ReplayBuffer,
    max_episode_length,
    action_fn,
    render=False,
    reward_per_agent=False,
):
    obs = env.reset()
    dones = [False] * env.n_agents

    episode_steps = 0
    episode_return = 0

    while not any(dones):
        if (render): env.render()

        acts = action_fn(obs)
        nobs, rwds, dones, _ = env.step(np.array(acts))

        episode_steps += 1
        if (episode_steps >= max_episode_length): # Some envs don't have done flags,
            dones = [True] * env.n_agents #  so manually set them here

        buffer.store(
            obs=obs,
            acts=acts,
            rwds=rwds,
            nobs=nobs,
            dones=dones,
        )
        episode_return += rwds[0] if reward_per_agent else sum(rwds)
        obs = nobs

    return episode_return, episode_steps

def train(config: argparse.Namespace):
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = create_env(config.env)
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=10e6,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size,
    )

    gradient_estimator = ...
    match config.gradient_estimator: # TODO: Pass parameters better
        case "stgs":
            gradient_estimator = gradient_estimators.STGS(config.gumbel_temp)
        case "grmck":
            gradient_estimator = gradient_estimators.GRMCK(config.gumbel_temp, config.rao_k)
        case "gst":
            gradient_estimator = gradient_estimators.GST(config.gumbel_temp, config.gst_gap)
        case _:
            print("Unknown gradient estimator type")

    maddpg = MADDPG(
        env=env,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        gradient_clip=config.gradient_clip,
        hidden_dim_width=config.hidden_dim_width,
        gamma=config.gamma,
        gumbel_temp=config.gumbel_temp,
        policy_regulariser=config.policy_regulariser,
        gradient_estimator=gradient_estimator,
        standardise_rewards=config.standardise_rewards,
    )

    # Warm up:
    for _ in tqdm(range(config.warmup_episodes), bar_format=BAR_FORMAT, postfix="Warming up..."):
        _, _ = play_episode(
            env,
            buffer,
            max_episode_length=config.max_episode_length,
            action_fn=(lambda _ : env.action_space.sample()),
        )

    eval_returns = []
    with tqdm(total=config.total_steps, bar_format=BAR_FORMAT) as pbar:
        elapsed_steps = 0
        eval_count = 0
        while elapsed_steps < config.total_steps:
            _, episode_steps = play_episode(
                env,
                buffer,
                max_episode_length=config.max_episode_length,
                action_fn=maddpg.acts,
                render=False,
            )

            if (not config.disable_training):
                sample = buffer.sample()
                maddpg.update(sample)

            if (config.eval_freq != 0 and (eval_count * config.eval_freq) <= elapsed_steps):
                eval_count += 1

                timestep_returns = []
                for _ in range(config.eval_iterations):
                    timestep_returns.append(play_episode(
                            env,
                            buffer,
                            max_episode_length=config.max_episode_length,
                            action_fn=maddpg.acts,
                            reward_per_agent=config.reward_per_agent,
                        )[0]
                    )
                
                eval_returns.append( np.mean(timestep_returns) )
                pbar.set_postfix(eval_return=f"{np.round(np.mean(timestep_returns), 2)}", refresh=True)
                wandb.log({f"Return": np.mean(timestep_returns)}, step=elapsed_steps)

                if config.render:
                    play_episode(
                        env,
                        buffer,
                        max_episode_length=config.max_episode_length,
                        action_fn=maddpg.acts,
                        render=True,
                    )

            elapsed_steps += episode_steps
            pbar.update(episode_steps)

    env.close()

    return eval_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--warmup_episodes", default=400, type=int)
    parser.add_argument("--total_steps", default=2_000_000, type=int)
    parser.add_argument("--max_episode_length", default=25, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden_dim_width", default=64, type=int) # TODO: Pass this as a tuple and unpack into n layers in network creation?
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--eval_freq", default=25_000, type=int)
    parser.add_argument("--eval_iterations", default=100, type=int)
    parser.add_argument("--gumbel_temp", default=1.0, type=float)
    parser.add_argument("--rao_k", default=1, type=int)
    parser.add_argument("--gst_gap", default=1.0, type=float)
    parser.add_argument("--gradient_estimator", default="stgs", choices=[
        "stgs",
        "grmck",
        "gst",
    ], type=str)
    parser.add_argument("--policy_regulariser", default=0.001, type=float)
    parser.add_argument("--reward_per_agent", action="store_true")
    parser.add_argument("--standardise_rewards", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_training", action="store_true")
    parser.add_argument("--wandb_project_name", default="maddpg-sink-project", type=str)
    parser.add_argument("--disable_wandb", action="store_true")

    config = parser.parse_args()

    run = wandb.init(
        project=config.wandb_project_name,
        name=f"{str(date.today())}-{config.env}-{config.seed}",
        entity="callumtilbury",
        mode="disabled" if (config.disable_wandb) else "online",
    )
    wandb.config.update(config)

    _ = train(config)
