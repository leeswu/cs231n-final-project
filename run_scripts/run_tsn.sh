#!/bin/bash

# Run training
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics_dance.py

# Run Testing
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics_dance.py \
    work_dirs/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics_dance.py/best_acc_*.pth

# Shut down the VM after the training is complete
sudo shutdown -h now