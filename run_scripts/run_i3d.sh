#!/bin/bash

# Run training
python tools/train.py ../configs/recognition/i3d/i3d_dance.py

# Run Testing
python tools/test.py ../configs/recognition/i3d/i3d_dance.py \
    ../work_dirs/i3d_dance.py/best_acc_*.pth

# Shut down the VM after the training is complete
sudo shutdown -h now