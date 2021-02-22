# coding=utf-8

"""
1. 对指定目录下的模型进行统计
2. 对排序后的每个模型依次进行:
   2.1 调用 test_kitti_depth.py 生成 npy 文件
   2.2 调用 kitti_eval/eval_depth.py 生成性能指标
   2.3 将 性能结果写入 txt 及 xls 文件
"""

import sys
import os
import os.path as osp
import cv2
import numpy as np
import datetime
import time
import logging
import subprocess

from configs import cfg as gcfg


def get_model_list():
    logging.info('get_model_list()')
    model_dir = gcfg.get_in_model_dir
    models = dict()
    for item in os.listdir(model_dir):
        if item.endswith('.index'):
            model_name, _ = osp.splitext(item)
            file_meta = osp.join(model_dir, model_name + '.meta')
            file_data = osp.join(model_dir, model_name + '.data-00000-of-00001')
            if not osp.exists(file_meta) or not osp.exists(file_data):
                logging.error('uncomplete model: {}'.format(item))
                raise ValueError

            tags = model_name.split('-')
            key_step = '{:010d}'.format(np.int32(tags[1]))
            models[key_step] = osp.join(model_dir, model_name)
    keys = list(models.keys())
    keys.sort()
    models_sorted = list()
    for k in keys:
        print('{}'.format(osp.basename(models[k])))
        models_sorted.append((k, models[k]))
    return models_sorted


def auto_test_and_eval_pipeline():
    logging.info('auto_test_and_eval_pipeline()')
    models = get_model_list()
    for step_tag, model_name in models:
        print('process model: {}'.format(model_name))

        # run test
        '''
        subprocess.call(['python', 'test_kitti_depth.py',
                         '--dataset_dir', '/disk4t0/0-MonoDepth-Database/KITTI_FULL/',
                         '--output_dir', 'kitti_eval/',
                         '--batch_size', 1,
                         '--is_sfmlearner', 0,
                         '--ckpt_file_sfm', 'models/model-190532',
                         '--ckpt_file_depth', model_name])
        '''

        # run eval
        break


################################################################################
# main
################################################################################
if __name__ == '__main__':
    print('sys.version:     {}'.format(sys.version))
    print('np.__version__:  {}'.format(np.__version__))
    print('cv2.__version__: {}'.format(cv2.__version__))
    print('start @{}'.format(datetime.datetime.now()))
    time_beg = time.time()
    auto_test_and_eval_pipeline()
    time_end = time.time()
    print('elapsed {} seconds.'.format(time_end - time_beg))
