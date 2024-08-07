#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

fix_w=128
lambda_nse=10.0
noise_emb_type=BEATs
inject_type=FiLM
inject_layers=11,12,13,14,15,16,17,18,19,20
num_inject_layers=10

name=exp_fix_w_"$fix_w"_loss_nse_"$lambda_nse"_"$noise_emb_type"_"$inject_type"_"$num_inject_layers"

python train.py --dataroot ../VoiceBank-DEMAND/trainset_for_UNA-GAN --fix_w "$fix_w" --name "$name" --lambda_nse "$lambda_nse" --noise_emb_type "$noise_emb_type" --inject_type "$inject_type" --inject_layers "$inject_layers" --num_inject_layers "$num_inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints
python test.py --fix_w "$fix_w" --name "$name" --lambda_nse "$lambda_nse" --noise_emb_type "$noise_emb_type" --inject_type "$inject_type" --inject_layers "$inject_layers" --num_inject_layers "$num_inject_layers" --CUT_mode CUT --checkpoints_dir checkpoints --state Test
