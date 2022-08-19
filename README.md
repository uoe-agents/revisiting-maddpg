# Revisiting Discrete Gradient Estimation in MADDPG

Exploration of alternative gradient estimation techniques in MADDPG.

Repository link: <https://github.com/callumtilbury/revisiting-maddpg/>

Usage:
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
