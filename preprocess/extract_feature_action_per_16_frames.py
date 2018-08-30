# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import glob
import json

import cv2
import parse
from tqdm import trange

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
FLAGS = flags.FLAGS

"""
acls = ['ApplyEyeMakeup','ApplyLipstick','Archery','BabyCrawling','BalanceBeam','BandMarching','BaseballPitch',
        'Basketball','BasketballDunk','BenchPress','Biking', 'Billiards', 'BlowDryHair','BlowingCandles','BodyWeightSquats',
        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling',
        'CricketShot', 'CuttingInKitchen','Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics','FrisbeeCatch', 'FrontCrawl',
        'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace',
        'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump',
        'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello','PlayingDaf',
        'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault',
        'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard',
        'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling',
        'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars',
        'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']
"""

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=None)
    return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var

def run_test():
    model_name = "./sports1m_finetuning_ucf101.model"

    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    with tf.variable_scope('var_name') as var_scope:
        weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
                }
        biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
                }

    # batch_size: None, NUM_FRAMES_PER_CLIP, CROP_SIZE, CROP_SIZE, CHANNELS
    logits, video_feature_vector = c3d_model.inference_c3d(images_placeholder, 1, FLAGS.batch_size, weights, biases)

    norm_score = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)

    # test file
    # test_list_file = 'list/test_ucf101.list'
    test_list_file = 'friends.list'
    with open(test_list_file, 'r') as fin:
        frows = fin.readlines()
        test_video_dpaths = list(map(lambda r: r.strip('\n').split()[0], frows))
    print("Number of test videos = {}".format(len(test_video_dpaths)))

    # mean_values of UCF-101
    np_mean = np.load('crop_mean.npy').reshape([c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, 3]) # TODO

    out_dpath = "./data/feature_action"
    os.makedirs(out_dpath, exist_ok=True)
    video_meta_parser = parse.compile("{season:d}x{episode:d}_{:w}")
    for video_dpath in test_video_dpaths:
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.

        video_fname = video_dpath.split('/')[-1]
        video_meta = video_meta_parser.parse(video_fname)
        with open('./data/actions/S{:02d}_E{:02d}.json'.format(video_meta["season"], video_meta["episode"])) as fin:
            video_action_labels = json.load(fin)

        # number of frames
        num_of_frames = len(glob.glob(video_dpath + '/*'))

        frame_action_list = []
        pbar = trange(0, num_of_frames - c3d_model.NUM_FRAMES_PER_CLIP)
        for frame in pbar:
            action_label = video_action_labels.get(str(frame), None)
            if action_label is None:
                continue

            pbar.set_description("Processing {} - Frame #{} ({})".format(video_fname, frame+1, ", ".join(action_label)))

            test_images, test_images_original = input_data.read_frames(video_dpath, frame, np_mean)

            # expand dim [ 1 x batch_size x clips x crop x crop x 3]
            test_images = np.expand_dims(test_images, 0)

            predict_score = norm_score.eval(session=sess, feed_dict={ images_placeholder: test_images })
            video_feature = video_feature_vector.eval(session=sess, feed_dict={ images_placeholder: test_images }).squeeze(0)

            frame_action_dict = {
                "frame": frame,
                "feature_vector": video_feature,
                "actions": action_label
            }
            frame_action_list.append(frame_action_dict)
        np.save('{}/{}'.format(out_dpath, video_fname), frame_action_list)


def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
