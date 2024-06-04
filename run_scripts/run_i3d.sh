#!/bin/bash

# Run training
python tools/train.py configs/recognition/i3d/i3d_i3dpl_dance.py
# Run Testing
python tools/test.py configs/recognition/i3d/i3d_i3dpl_dance.py \
    work_dirs/i3d_i3dpl_dance/best_acc_*.pth

# Run training
python tools/train.py configs/recognition/i3d/i3d_i3dpl_heavy_dance.py
# Run Testing
python tools/test.py configs/recognition/i3d/i3d_i3dpl_heavy_dance.py \
    work_dirs/i3d_i3dpl_heavy_dance/best_acc_*.pth

# Run training
python tools/train.py configs/recognition/i3d/i3d_tsnpl_dance.py
# Run Testing
python tools/test.py configs/recognition/i3d/i3d_tsnpl_dance.py \
    work_dirs/i3d_tsnpl_dance/best_acc_*.pth

# Run training
python tools/train.py configs/recognition/i3d/i3d_tsnpl_heavy_dance.py
# Run Testing
python tools/test.py configs/recognition/i3d/i3d_tsnpl_heavy_dance.py \
    work_dirs/i3d_tsnpl_heavy_dance/best_acc_*.pth


# Shut down the VM after the training is complete
sudo shutdown -h now