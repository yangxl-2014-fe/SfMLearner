#!/bin/bash

: '
conda activate onnx
cd /home/ftx/Documents/yangxl-2014-fe/my_forked/SfMLearner

bash run_test.sh

model-42260.index
model-63390.index
model-84520.index
model-105650.index
model-126780.index
model-147910.index
model-169040.index
model-190170.index
model-211300.index
model-21130.index
model-232430.index
model-253560.index
model-274690.index
model-295820.index
model-316950.index
model-338080.index
model-359210.index
'

depth_model_dir=/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/checkpoints_depth/mytrain_kitti_learned_intrinsics-archive

python test_kitti_depth.py \
  --dataset_dir /disk4t0/0-MonoDepth-Database/KITTI_FULL/ \
  --output_dir kitti_eval/ \
  --batch_size 1 \
  --is_sfmlearner 0 \
  --ckpt_file_sfm models/model-190532 \
  --ckpt_file_depth  ${depth_model_dir}/model-359210

