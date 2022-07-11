import argparse
import einops
import torch
from tqdm import tqdm; BAR_FORMAT = "{l_bar}{bar:50}{r_bar}{bar:-10b}"
from buffer import ReplayBuffer
import numpy as np
import scipy.stats as st
from env_wrapper import create_env
from maddpg import MADDPG
import wandb
from datetime import date

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

def train(config: argparse.Namespace, seed : int):
    # Set seeds
    torch.manual_seed(ss)
    np.random.seed(ss)

    env = create_env(config.env)
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=10e6,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size,
    )

    maddpg = MADDPG(
        env=env,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        gradient_clip=config.gradient_clip,
        hidden_dim_width=config.hidden_dim_width,
        gamma=config.gamma,
        gumbel_temp=config.gumbel_temp,
        policy_regulariser=config.policy_regulariser,
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
                wandb.log({f"Return": np.mean(timestep_returns)}, step=elapsed_steps, commit=False)

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
    parser.add_argument("--n_seeds", default=5, type=int)
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
    parser.add_argument("--policy_regulariser", default=0.001, type=float)
    parser.add_argument("--reward_per_agent", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_training", action="store_true")
    parser.add_argument("--wandb_project_name", default="maddpg-sink-project", type=str)
    parser.add_argument("--disable_wandb", action="store_true")

    config = parser.parse_args()

    eval_returns_across_seeds = []
    for ss in range(config.n_seeds):
        run = wandb.init(
            project=config.wandb_project_name,
            name=f"{str(date.today())}-{config.env}-{ss}",
            entity="callumtilbury",
            group=f"{config.env}",
            reinit=True,
            mode="disabled" if (config.disable_wandb) else "online",
        )
        wandb.config.update(config)

        eval_returns = train(config, ss)
        eval_returns_across_seeds.append(eval_returns)

    # MAX stat
    average_eval_returns_across_seeds = np.mean(eval_returns_across_seeds, axis=0)
    max_timestep = np.argmax(average_eval_returns_across_seeds)
    eval_returns_across_seeds_max_timestep = np.array(eval_returns_across_seeds)[:,max_timestep]
    max_mean = np.mean(eval_returns_across_seeds_max_timestep)
    max_95perc = st.t.interval(
        alpha=0.95,
        df=config.n_seeds - 1,
        loc=max_mean,
        scale=st.sem(eval_returns_across_seeds_max_timestep)
    )[0]

    print(f"Max: {max_mean} ± {np.abs(max_mean - max_95perc)}")
    wandb.run.summary["Max Return (mean)"] = max_mean
    wandb.run.summary["Max Return (err)"] = np.abs(max_mean - max_95perc)

    # AVERAGE stat
    flattened_eval_returns = einops.rearrange(np.array(eval_returns_across_seeds), 'seeds values -> (seeds values)')
    avg_mean = np.mean(flattened_eval_returns)
    avg_95perc = st.t.interval(
        alpha=0.95,
        df=len(flattened_eval_returns) - 1,
        loc=avg_mean,
        scale=st.sem(flattened_eval_returns)
    )[0]

    print(f"Avg: {avg_mean} ± {np.abs(avg_mean - avg_95perc)}")
    wandb.run.summary["Avg Return (mean)"] = avg_mean
    wandb.run.summary["Avg Return (err)"] = np.abs(avg_mean - avg_95perc)
