#!/bin/bash

: '
conda activate onnx
cd /home/ftx/Documents/yangxl-2014-fe/my_forked/SfMLearner
'

python kitti_eval/eval_depth.py \
      --kitti_dir=/disk4t0/0-MonoDepth-Database/KITTI_FULL/ \
      --pred_file=kitti_eval/kitti_eigen_depth_predictions.npy