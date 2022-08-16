import argparse
from typing import List
import torch
from tqdm import tqdm; BAR_FORMAT = "{l_bar}{bar:50}{r_bar}{bar:-10b}"
from agent import Agent
from buffer import ReplayBuffer
import numpy as np
import scipy.stats as st
from env_wrapper import create_env
from maddpg import MADDPG
import wandb
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib import RunDisabled
from datetime import date
from time import time, sleep
import gradient_estimators
import yaml
import os.path as path

def play_episode(
    env,
    buffer : ReplayBuffer | None,
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
        if (render):
            env.render()
            sleep(0.03)

        acts = action_fn(obs)
        nobs, rwds, dones, _ = env.step(np.array(acts))

        episode_steps += 1
        if (episode_steps >= max_episode_length): # Some envs don't have done flags,
            dones = [True] * env.n_agents #  so manually set them here

        if buffer is not None:
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

def train(config: argparse.Namespace, wandb_run: Run | RunDisabled | None):
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if (config.env == ""):
        print("Environment not set!")
        return

    env = create_env(config.env)
    observation_dims = np.array([obs.shape[0] for obs in env.observation_space])
    buffer = ReplayBuffer(
        capacity=config.replay_buffer_size,
        obs_dims=observation_dims, # TODO: change format of the replay buffer input??
        batch_size=config.batch_size,
    )

    gradient_estimator = ...
    match config.gradient_estimator:
        case "stgs":
            gradient_estimator = gradient_estimators.STGS(config.gumbel_temp)
        case "grmck":
            gradient_estimator = gradient_estimators.GRMCK(config.gumbel_temp, config.rao_k)
        case "gst":
            gradient_estimator = gradient_estimators.GST(config.gumbel_temp, config.gst_gap)
        case "tags":
            gradient_estimator = gradient_estimators.TAGS(config.tags_start, config.tags_end, config.tags_period)
        case _:
            print("Unknown gradient estimator type")
            return None

    pretrained_agents = None if config.pretrained_agents == "" \
        else torch.load(config.pretrained_agents)

    maddpg = MADDPG(
        env=env,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        gradient_clip=config.gradient_clip,
        hidden_dim_width=config.hidden_dim_width,
        gamma=config.gamma,
        soft_update_size=config.soft_update_size,
        policy_regulariser=config.policy_regulariser,
        gradient_estimator=gradient_estimator,
        standardise_rewards=config.standardise_rewards,
        pretrained_agents=pretrained_agents,
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
                for _ in range(config.train_repeats):
                    sample = buffer.sample()
                    if sample is not None:
                        maddpg.update(sample)
                if config.log_grad_variance and elapsed_steps % config.log_grad_variance_interval == 0:
                    for agent in maddpg.agents:
                        for name, param in agent.policy.named_parameters():
                            wandb.log({
                                f"{agent.agent_idx}-{name}-grad" : torch.var(param.grad).item(),
                            }, step=elapsed_steps)

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
                wandb.log({
                    f"Return": np.mean(timestep_returns),
                    f"Timestep": int(time()),
                }, step=elapsed_steps)

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

    if config.save_agents:
        save_path = path.join("saved_agents",f"maddpg_{config.env}_{int(time())}.pt")
        torch.save(maddpg.agents, save_path)

        artifact = wandb.Artifact(name=f"{config.env}_agents", type="agents")
        artifact.add_file(save_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return eval_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Option to override params from config file
    parser.add_argument("--config_file", default="", type=str)
    
    # Set env & seed
    parser.add_argument("--env", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)

    # Episode length etc.
    parser.add_argument("--warmup_episodes", default=400, type=int)
    parser.add_argument("--replay_buffer_size", default=2_000_000, type=int)
    parser.add_argument("--total_steps", default=2_000_000, type=int)
    parser.add_argument("--max_episode_length", default=25, type=int)
    parser.add_argument("--train_repeats", default=1, type=int)
    
    # Core hyperparams
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden_dim_width", default=64, type=int)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--soft_update_size", default=0.01, type=float)
    parser.add_argument("--policy_regulariser", default=0.001, type=float)
    parser.add_argument("--reward_per_agent", action="store_true")
    parser.add_argument("--standardise_rewards", action="store_true")

    # When to evaluate performance
    parser.add_argument("--eval_freq", default=50_000, type=int)
    parser.add_argument("--eval_iterations", default=100, type=int)

    # Gradient Estimation hyperparams
    parser.add_argument("--gradient_estimator", default="stgs", choices=[
        "stgs",
        "grmck",
        "gst",
        "tags",
    ], type=str)
    parser.add_argument("--gumbel_temp", default=1.0, type=float)
    parser.add_argument("--rao_k", default=1, type=int) # For GRMCK
    parser.add_argument("--gst_gap", default=1.0, type=float) # For GST
    parser.add_argument("--tags_start", default=5.0, type=float) # For TAGS
    parser.add_argument("--tags_end", default=0.5, type=float) # For TAGS
    parser.add_argument("--tags_period", default=2_000_000, type=int) # For TAGS
    
    # Ability to save & load agents
    parser.add_argument("--save_agents", action="store_true")
    parser.add_argument("--pretrained_agents", default="", type=str)
    parser.add_argument("--just_demo_agents", action="store_true")

    # Misc
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_training", action="store_true")
    
    # WandB
    parser.add_argument("--wandb_project_name", default="maddpg-sink-project", type=str)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--offline_wandb", action="store_true")
    parser.add_argument("--log_grad_variance", action="store_true")
    parser.add_argument("--log_grad_variance_interval", default=1000, type=int)

    config = parser.parse_args()

    # TODO: This is a bit gimmicky, but works for now
    if (config.just_demo_agents and config.pretrained_agents != ""):
        env = create_env(config.env)
        agents : List[Agent] = torch.load(config.pretrained_agents)
        maddpg = MADDPG(
            env=env,
            pretrained_agents=agents,
            # ———
            # Not needed for demoing
            critic_lr=0,
            actor_lr=0,
            gradient_clip=0,
            hidden_dim_width=0,
            gamma=0,
            soft_update_size=0,
            policy_regulariser=0,
            gradient_estimator=None,
            standardise_rewards=False,
            # ———
        )
        for _ in range(100):
            play_episode(
                env=env,
                buffer=None,
                max_episode_length=config.max_episode_length,
                action_fn=maddpg.acts,
                render=True,
            )
            if (input() == "e"): # Pause between renders
                break
        env.close()
        exit(0)

    if (config.config_file != ""):
        with open(config.config_file) as cf:
            if 'base' in (yaml_config := yaml.load(cf, Loader=yaml.FullLoader)):
                with open(path.join(path.dirname(config.config_file), yaml_config['base'])) as cf_base:
                    vars(config).update( yaml.load(cf_base, Loader=yaml.FullLoader) )
            vars(config).update( yaml_config ) # Child takes update preference

    wandb_mode = "online"
    if config.disable_wandb: # Disabling takes priority
        wandb_mode = "disabled"
    elif config.offline_wandb: # Don't sync to wandb servers during run
        wandb_mode = "offline"

    wandb_run = wandb.init(
        project=config.wandb_project_name,
        name=f"{str(date.today())}-{config.env}-{config.seed}",
        entity="callumtilbury",
        mode=wandb_mode,
    )
    wandb.config.update(config)

    _ = train(config, wandb_run)
