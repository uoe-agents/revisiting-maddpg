#!/bin/sh

echo "MPE [A]: Simple Speaker Listener"
python main.py \
    --env simple_speaker_listener \
    --critic_lr 5e-4 \
    --actor_lr 5e-4 \
    --policy_regulariser 0.01 \
    --reward_per_agent \
    --wandb_project_name maddpg-baselines \

echo "MPE [B]: Simple Spread"
python main.py \
    --env simple_spread \
    --critic_lr 5e-4 \
    --actor_lr 5e-4 \
    --policy_regulariser 0.01 \
    --reward_per_agent \
    --wandb_project_name maddpg-baselines \

echo "LBF [A]: Foraging-8x8-2p-2f-c"
python main.py \
    --env Foraging-8x8-2p-2f-coop-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [B]: Foraging-8x8-2p-2f-2s-c"
python main.py \
    --env Foraging-8x8-2p-2f-2s-c-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [C]: Foraging-10x10-3p-3f-v2"
python main.py \
    --env Foraging-10x10-3p-3f-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [D]: Foraging-10x10-3p-3f-2s-v2"
python main.py \
    --env Foraging-10x10-3p-3f-2s-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [E]: Foraging-15x15-3p-5f-v2"
python main.py \
    --env Foraging-15x15-3p-5f-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [F]: Foraging-15x15-4p-3f-v2"
python main.py \
    --env Foraging-15x15-4p-3f-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "LBF [G]: Foraging-15x15-4p-5f-v2"
python main.py \
    --env Foraging-15x15-4p-5f-v2 \
    --critic_lr 3e-4 \
    --actor_lr 3e-4 \
    --policy_regulariser 0.001 \
    --wandb_project_name maddpg-baselines \

echo "RWARE [A]: tiny-2p"
python main.py \
    --env rware-tiny-2ag-v1 \
    --critic_lr 5e-4 \
    --actor_lr 5e-4 \
    --policy_regulariser 0.001 \
    --total_steps 4_000_000 \
    --wandb_project_name maddpg-baselines \

echo "RWARE [B]: tiny-4p"
python main.py \
    --env rware-tiny-4ag-v1 \
    --critic_lr 5e-4 \
    --actor_lr 5e-4 \
    --policy_regulariser 0.001 \
    --total_steps 4_000_000 \
    --wandb_project_name maddpg-baselines \

echo "RWARE [C]: small-4p"
python main.py \
    --env rware-small-4ag-v1 \
    --critic_lr 5e-4 \
    --actor_lr 5e-4 \
    --policy_regulariser 0.001 \
    --total_steps 4_000_000 \
    --wandb_project_name maddpg-baselines \
