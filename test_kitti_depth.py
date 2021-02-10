# coding=utf-8

from __future__ import division
import tensorflow as tf
import numpy as np
import os
import sys
import os.path as osp
import logging
# import scipy.misc
import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt

sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '..'))
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)), '../google-research'))
print('sys.path:')
for item in sys.path:
    print('  - {}'.format(item))

# User import
from configs import cfg as gcfg

from SfMLearner import SfMLearner
# https://stackoverflow.com/a/42121886
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from depth_from_video_in_the_wild import model

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file_sfm", None, "checkpoint file of SfMLearner")
flags.DEFINE_string("ckpt_file_depth", None, "checkpoint file of depth from video")
flags.DEFINE_integer("is_sfmlearner", 0, "SfMLearner / depth_from_video")
FLAGS = flags.FLAGS


def main(_):
    # logging
    logging.info('Settings:')
    logging.info('  - batch_size:      {}'.format(FLAGS.batch_size))
    logging.info('  - img_height:      {}'.format(FLAGS.img_height))
    logging.info('  - img_width:       {}'.format(FLAGS.img_width))
    logging.info('  - dataset_dir:     {}'.format(FLAGS.dataset_dir))
    logging.info('  - output_dir:      {}'.format(FLAGS.output_dir))
    logging.info('  - ckpt_file_sfm:   {}'.format(FLAGS.ckpt_file_sfm))
    logging.info('  - ckpt_file_depth: {}'.format(FLAGS.ckpt_file_depth))
    logging.info('  - is_sfmlearner:   {}'.format(FLAGS.is_sfmlearner))

    with open('data/kitti/test_files_eigen.txt', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # save name
    if FLAGS.is_sfmlearner:
        basename = os.path.basename(FLAGS.ckpt_file_sfm)
    else:
        basename = os.path.basename(FLAGS.ckpt_file_depth)
    output_file = FLAGS.output_dir + '/' + basename

    if FLAGS.is_sfmlearner:
        # model: SfMLearner
        sfm = SfMLearner()
        sfm.setup_inference(img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            batch_size=FLAGS.batch_size,
                            mode='depth')

        saver = tf.train.Saver([var for var in tf.model_variables()])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.ckpt_file_sfm)
            pred_all = []
            for t in range(0, len(test_files), FLAGS.batch_size):
                if t % 100 == 0:
                    print('processing %s: %d/%d' % (basename, t, len(test_files)))
                inputs = np.zeros(
                    (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                    dtype=np.uint8)
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    '''
                    fh = open(test_files[idx], 'r')
                    raw_im = pil.open(fh)
                    '''
                    raw_im = pil.open(test_files[idx])
                    scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                    inputs[b] = np.array(scaled_im)
                    # im = scipy.misc.imread(test_files[idx])
                    # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
                pred = sfm.inference(inputs, sess, mode='depth')
                depth = pred['depth']
                # depth: <class 'numpy.ndarray'> (1, 128, 416, 1) float32
                for b in range(FLAGS.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    # pred_all.append(pred['depth'][b, :, :, 0])
                    pred_all.append(depth[b, :, :, 0])
            np.save(output_file, pred_all)
    else:
        # model: depth_from_video_in_the_wild
        depth_model = model.Model(img_height=None, img_width=None,
                                  batch_size=None,
                                  is_training=False)
        depth_sess = tf.Session()
        depth_saver = tf.train.Saver()
        depth_saver.restore(depth_sess, FLAGS.ckpt_file_depth)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                '''
                fh = open(test_files[idx], 'r')
                raw_im = pil.open(fh)
                '''
                raw_im = pil.open(test_files[idx])
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height),
                                          pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            depth = depth_model.inference_depth(inputs, depth_sess)
            # depth: <class 'numpy.ndarray'> (1, 128, 416, 1) float32
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                # pred_all.append(pred['depth'][b, :, :, 0])
                pred_all.append(depth[b, :, :, 0])
        np.save(output_file, pred_all)


def infer_depth_via_depth_from_video_in_the_wild():
    logging.warning('infer_depth_via_depth_from_video_in_the_wild()')
    # Depth from Video in the Wild
    sys.path.append(
        '/home/ftx/Documents/yangxl-2014-fe/my_forked/google-research')
    # https://stackoverflow.com/a/42121886
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from depth_from_video_in_the_wild import model

    str_img = '/disk4t0/0-MonoDepth-Database/KITTI_FULL/2011_09_26/' \
              '2011_09_26_drive_0001_sync/image_02/data/0000000000.png'
    str_dep = '/disk4t0/0-MonoDepth-Database/_TMP_Infered_Depth/depth.png'
    image = cv2.imread(str_img)
    print('image: {} {} {}'.format(type(image), image.shape, image.dtype))

    train_model = model.Model(img_height=None, img_width=None, batch_size=None,
                              is_training=False)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,
                  '/disk4t0/0-MonoDepth-Database/depth_from_video_in_the_wild/'
                  'checkpoints_depth/kitti_learned_intrinsics/model-248900')
    inputs = cv2.resize(image, (416, 128))[np.newaxis, :]
    depth = train_model.inference_depth(inputs, sess)

    print('\n\n{}'.format('=' * 60))
    print('depth: {} {} {}\n\n'.format(type(depth), depth.shape, depth.dtype))
    # depth: <class 'numpy.ndarray'> (1, 128, 416, 1) float32

    dd = (depth[0] * 256)
    cv2.imwrite(str_dep, dd)

    plt.figure(figsize=(20, 10))
    plt.imshow(image)
    plt.figure(figsize=(20, 10))
    plt.imshow(1 / depth[0, :, :, 0], cmap='plasma')


def infer_depth_v2():
    logging.warning('infer_depth_v2()')

    pass


################################################################################
# main
################################################################################
if __name__ == '__main__':
    tf.app.run()
