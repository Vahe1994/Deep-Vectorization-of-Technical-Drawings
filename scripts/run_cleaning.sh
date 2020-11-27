#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_cleaning.py --model 'UNET' --n_epochs 30 --datadir '/dataset/Cleaning/Synthetic/train' --valdatadir '/dataset/Cleaning/Synthetic/val' --batch_size 4