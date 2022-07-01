import argparse
from tqdm import tqdm
from buffer import ReplayBuffer
import numpy as np
from env_wrapper import create_env
from maddpg import MADDPG
import wandb

def play_episode(
    env,
    buffer : ReplayBuffer,
    max_timesteps_per_episode,
    action_fn,
    render=False,
):
    obs = env.reset()
    dones = [False] * env.n_agents

    steps = 0
    episode_return = 0

    while not any(dones):# and (steps < max_timesteps_per_episode):
        if (render): env.render()

        acts = action_fn(obs)#maddpg.acts(obs)
        nobs, rwds, dones, _ = env.step(np.array(acts))

        steps += 1
        if (steps >= max_timesteps_per_episode):
            dones = [True] * env.n_agents

        buffer.store(
            obs=obs,
            acts=acts,
            rwds=rwds,
            nobs=nobs,
            dones=dones,
        )
        episode_return += sum(rwds)
        obs = nobs

    return episode_return

def train(config: argparse.Namespace):
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
    )

    # Warm up:
    for _ in tqdm(range(config.warmup_episodes)):
        play_episode(
            env,
            buffer,
            max_timesteps_per_episode=config.episode_length,
            action_fn=(lambda _ : env.action_space.sample()),
        )

    print(f"Warmed up with {buffer.entries} entries")

    with tqdm(range(config.n_episodes),
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}") as pbar:
        for epi_i in pbar:
            _ = play_episode(
                env,
                buffer,
                max_timesteps_per_episode=config.episode_length,
                action_fn=maddpg.acts,
                render=False,
            )

            if (not config.disable_training):
                sample = buffer.sample()
                maddpg.update(sample)

            if (config.eval_freq != 0 and epi_i % config.eval_freq == 0):
                eval_returns = []
                for _ in range(config.eval_iterations):
                    eval_returns.append(play_episode(
                        env,
                        buffer,
                        max_timesteps_per_episode=config.episode_length,
                        action_fn=maddpg.acts,
                        render=config.render,
                    ))
                pbar.set_postfix(eval_return=f"{np.round(np.mean(eval_returns), 2)} ({np.round(np.std(eval_returns),2)})", refresh=True)
                wandb.log({"Ep. Return (Eval)": np.mean(eval_returns)})

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_speaker_listener")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--warmup_episodes", default=400, type=int)
    #parser.add_argument("--timesteps", default=25000, type=int) # TODO?
    parser.add_argument("--episode_length", default=25, type=int)
    #parser.add_argument("--steps_per_update", default=100, type=int) # TODO: Bring back?
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden_dim_width", default=64, type=int) # TODO: Pass this as a tuple and unpack into n layers in network creation?
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--eval_iterations", default=100, type=int)
    parser.add_argument("--gumbel_temp", default=1.0, type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_training", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")

    config = parser.parse_args()

    wandb.init(
        project="maddpg-pytorch-first-tests", # TODO: parse as argument??
        entity="callumtilbury",
        mode="disabled" if (config.disable_wandb) else "online"
    )
    wandb.config = config

    train(config)
