#!/bin/bash

: '
conda activate onnx
cd /home/ftx/Documents/yangxl-2014-fe/my_forked/SfMLearner

bash run_test.sh

model-21130
model-42260
model-63390
model-84520
model-105650
model-126780
model-147910
model-169040
model-190170
model-211300
model-232430
model-253560
model-274690
model-295820
model-316950
model-338080
model-359210
model-697290
model-718420
model-739550
model-760680
model-781810
model-908590
model-929720
model-950850
model-971980
model-993110
'

depth_model_dir=/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/checkpoints_depth/mytrain_kitti_learned_intrinsics-archive

python test_kitti_depth.py \
  --dataset_dir /disk4t0/0-MonoDepth-Database/KITTI_FULL/ \
  --output_dir tmp/ \
  --batch_size 1 \
  --is_sfmlearner 0 \
  --ckpt_file_sfm models/model-190532 \
  --ckpt_file_depth  ${depth_model_dir}/model-84520

