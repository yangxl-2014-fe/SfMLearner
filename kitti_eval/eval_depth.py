# coding=utf-8

from __future__ import division
import sys
import datetime
import time
import cv2
import os
import os.path as osp
import numpy as np
import argparse
import logging

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))
print('sys.path:')
for item in sys.path:
    print('  - {}'.format(item))

# User import
from configs import cfg as gcfg
from kitti_eval.depth_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--test_file_list", type=str, default='./data/kitti/test_files_eigen.txt', 
                    help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()


################################################################################
# main
################################################################################
if __name__ == '__main__':
    print('sys.version:     {}'.format(sys.version))
    print('np.__version__:  {}'.format(np.__version__))
    print('cv2.__version__: {}'.format(cv2.__version__))
    print('start @{}\n\n'.format(datetime.datetime.now()))
    eval_beg = time.time()

    # logging
    logging.info('Settings:')
    logging.info('  - kitti_dir:      {}'.format(args.kitti_dir))
    logging.info('  - pred_file:      {}'.format(args.pred_file))
    logging.info('  - test_file_list: {}'.format(args.test_file_list))
    logging.info('  - min_depth:      {}'.format(args.min_depth))
    logging.info('  - max_depth:      {}'.format(args.max_depth))

    pred_depths = np.load(args.pred_file)
    test_files = read_text_lines(args.test_file_list)
    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(test_files, args.kitti_dir)

    num_test = len(im_files)

    # logging
    logging.info('Params:')
    logging.info('  - tested image numbers: {}'.format(num_test))
    logging.info('  - pred_depths:          {} {} {}'.format(type(pred_depths), pred_depths.shape, pred_depths.dtype))
    print('{} {} {}'.format('=' * 20, args.pred_file, '=' * 20))

    gt_depths = []
    pred_depths_resized = []
    for t_id in range(num_test):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        pred_depths_resized.append(
            cv2.resize(pred_depths[t_id], 
                       (im_sizes[t_id][1], im_sizes[t_id][0]), 
                       interpolation=cv2.INTER_LINEAR))
        depth = generate_depth_map(gt_calib[t_id], 
                                   gt_files[t_id], 
                                   im_sizes[t_id], 
                                   camera_id, 
                                   False, 
                                   True)
        gt_depths.append(depth.astype(np.float32))
    pred_depths = pred_depths_resized

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > args.min_depth, 
                              gt_depth < args.max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

    # time log
    eval_end = time.time()
    logging.warning('\n\n===> elaspsed {} seconds'.format(eval_end - eval_beg))
    logging.warning('done @{}'.format(datetime.datetime.now()))
    print('\n\n===> elaspsed {} seconds'.format(eval_end - eval_beg))
    print('done @{}'.format(datetime.datetime.now()))
