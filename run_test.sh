#!/bin/bash

source ./tools_yangxl2014fe.sh

# ==========
# time debug
# ==========
time_sh_start=$(date +"%s.%N")

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

depth_from_video_in_the_wild author provided:
model-248900
'
: '
depth_model_dir=/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/checkpoints_depth/mytrain_kitti_learned_intrinsics-archive
'

: '
depth_model_dir=/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/checkpoints_depth/2021.03.03-mytrain_kitti_learned_intrinsics

learner_choice:
  - 0: SfMLearner
  - 1: depth_from_video_in_the_wild
  - 2: depth_and_motion_learning
'

depth_from_video_in_the_wild_model_dir=/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/checkpoints_depth/kitti_learned_intrinsics
depth_and_motion_learning_model_dir=/disk4t0/0-MonoDepth-Database/Home-depth_and_motion_learning_no_mask

python test_kitti_depth.py \
  --dataset_dir /disk4t0/0-MonoDepth-Database/KITTI/ \
  --output_dir tmp/ \
  --batch_size 1 \
  --learner_choice 2 \
  --ckpt_file_sfm models/model-190532 \
  --ckpt_file_depth_from_video_in_the_wild  ${depth_from_video_in_the_wild_model_dir}/model-248900 \
  --ckpt_file_depth_and_motion_learning ${depth_and_motion_learning_model_dir}/model.ckpt-87311

# ==========
# time debug
# ==========
time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "run_test.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
