# Revisiting the Gumbel-Softmax in MADDPG

Exploration of alternative gradient estimation techniques in MADDPG.

## Hyperparameters

Hyperparameters used for the core MADDPG algorithm, mostly taken verbatim from [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869) by Papoudakis et al. (2021):
|  | LBF | RWARE |
| :--- | :---: | :---: |
| network type | MLP | MLP |
| hidden dimensions  | (64,64) | (64,64) |
| learning rate  | 3e-4 | 3e-4 |
| reward standardisation | True | True |
| policy regulariser | 0.001 | 0.001 |
| target update $\beta$ | 0.01 | 0.01 |
| max timesteps | 25 | 500 |
| training interval (steps) | 25 | 50 |

Hyperparameter details for the various gradient estimation techniques, with the chosen parameters listed for the two environments:
| Estimator: | Range Explored | LBF | RWARE |
| :--- | :---: | :---: | :---: |
| STGS-1 | $\tau=1.0$  | $1.0$ | $1.0$ |
| STGS-T | $\tau \in(0,1)$ | $0.5$ | $0.6$ |
| TAGS | $\tau \in[1,5] \rightarrow [0.1,0.5]$  | $4.0 \rightarrow 0.1$ | $1.0 \rightarrow 0.3$ |
| GRMCK | $\tau \in(0,1]; K= \{ 5,10,50 \}$ | $0.5;10$ | $0.7;5$ |
| GST | $\tau \in(0,1]$  | $0.7$ | $0.7$ |

## Code Usage
```
python main.py [-h] [--config_file CONFIG_FILE] [--env ENV] [--seed SEED] [--warmup_episodes WARMUP_EPISODES] [--replay_buffer_size REPLAY_BUFFER_SIZE]
               [--total_steps TOTAL_STEPS] [--max_episode_length MAX_EPISODE_LENGTH] [--train_repeats TRAIN_REPEATS] [--batch_size BATCH_SIZE]
               [--hidden_dim_width HIDDEN_DIM_WIDTH] [--critic_lr CRITIC_LR] [--actor_lr ACTOR_LR] [--gradient_clip GRADIENT_CLIP] [--gamma GAMMA]
               [--soft_update_size SOFT_UPDATE_SIZE] [--policy_regulariser POLICY_REGULARISER] [--reward_per_agent] [--standardise_rewards] [--eval_freq EVAL_FREQ]
               [--eval_iterations EVAL_ITERATIONS] [--gradient_estimator {stgs,grmck,gst,tags}] [--gumbel_temp GUMBEL_TEMP] [--rao_k RAO_K] [--gst_gap GST_GAP]
               [--tags_start TAGS_START] [--tags_end TAGS_END] [--tags_period TAGS_PERIOD] [--save_agents] [--pretrained_agents PRETRAINED_AGENTS] [--just_demo_agents]
               [--render] [--disable_training] [--wandb_project_name WANDB_PROJECT_NAME] [--disable_wandb] [--offline_wandb] [--log_grad_variance]
               [--log_grad_variance_interval LOG_GRAD_VARIANCE_INTERVAL]

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
  --env ENV
  --seed SEED
  --warmup_episodes WARMUP_EPISODES
  --replay_buffer_size REPLAY_BUFFER_SIZE
  --total_steps TOTAL_STEPS
  --max_episode_length MAX_EPISODE_LENGTH
  --train_repeats TRAIN_REPEATS
  --batch_size BATCH_SIZE
  --hidden_dim_width HIDDEN_DIM_WIDTH
  --critic_lr CRITIC_LR
  --actor_lr ACTOR_LR
  --gradient_clip GRADIENT_CLIP
  --gamma GAMMA
  --soft_update_size SOFT_UPDATE_SIZE
  --policy_regulariser POLICY_REGULARISER
  --reward_per_agent
  --standardise_rewards
  --eval_freq EVAL_FREQ
  --eval_iterations EVAL_ITERATIONS
  --gradient_estimator {stgs,grmck,gst,tags}
  --gumbel_temp GUMBEL_TEMP
  --rao_k RAO_K
  --gst_gap GST_GAP
  --tags_start TAGS_START
  --tags_end TAGS_END
  --tags_period TAGS_PERIOD
  --save_agents
  --pretrained_agents PRETRAINED_AGENTS
  --just_demo_agents
  --render
  --disable_training
  --wandb_project_name WANDB_PROJECT_NAME
  --disable_wandb
  --offline_wandb
  --log_grad_variance
  --log_grad_variance_interval LOG_GRAD_VARIANCE_INTERVAL
```
