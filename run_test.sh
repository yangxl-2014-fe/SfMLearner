#!/bin/bash

: '
conda activate onnx
cd /home/ftx/Documents/yangxl-2014-fe/my_forked/SfMLearner

bash run_test.sh
'

python test_kitti_depth.py \
  --dataset_dir /disk4t0/0-MonoDepth-Database/KITTI_FULL/ \
  --output_dir kitti_eval/ \
  --ckpt_file models/model-190532