#!/bin/bash

source ./tools_yangxl2014fe.sh

# ==========
# time debug
# ==========
time_sh_start=$(date +"%s.%N")

: '
conda activate onnx
cd /home/ftx/Documents/yangxl-2014-fe/my_forked/SfMLearner

bash run_eval.sh

model-190532.npy
download_kitti_eigen_depth_predictions.npy

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

python kitti_eval/eval_depth.py \
  --kitti_dir=/disk4t0/0-MonoDepth-Database/KITTI_FULL/ \
  --pred_file=tmp/model-190532.npy

# ==========
# time debug
# ==========
time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "run_eval.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
