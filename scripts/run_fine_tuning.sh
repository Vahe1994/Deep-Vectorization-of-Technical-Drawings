#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0 python main_cleaning.py --model 'UNET' --n_epochs 30 --datadir '/dataset/Cleaning/Real/train' --valdatadir '/dataset/Cleaning/Real/val' --batch_size 4 --name 'UNET_test'