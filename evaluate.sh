#!/bin/bash

# re10k testing
python evaluate.py \
    hydra.run.dir=/home/soohong/flash3d \
    hydra.job.chdir=true \
    +experiment=layered_re10k \
    +dataset.crop_border=false \
    dataset.test_split_path=splits/re10k_pixelsplat/test_one_diff_30.txt \
    model.depth.version=v1 \
    ++eval.save_vis=true

