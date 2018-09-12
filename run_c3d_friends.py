# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from datetime import datetime
from io import StringIO
import json
import math
import os
import time
import tempfile

import cv2
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import summary_op_util
import parse

import c3d_model
import input_data
from logger import Logger


# Basic model parameters as external flags.
flags = tf.app.flags
GPU_LIST = [1]
N_GPU = len(GPU_LIST)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([ str(i) for i in GPU_LIST ])
flags.DEFINE_integer('batch_size', 20, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
DATA_FPATH = "./list/friends_test_s01_e09.list"
BBOX_DPATH = "./data/friends_json/bbox/person"
SHOW_PERSON_BBOX=True

MODEL_NAME = "./models/friends/full/c3d_friends_model"
MODEL_STEP = 4999
MODEL_FPATH = "{}-{}".format(MODEL_NAME, MODEL_STEP)

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
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
        c3d_model.NUM_FRAMES_PER_CLIP,
        c3d_model.CROP_SIZE,
        c3d_model.CROP_SIZE,
        c3d_model.CHANNELS)
    )
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, c3d_model.NUM_CLASSES))
    return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(name_scope, logits, labels):
    """
    cross_entropy_mean = tf.reduce_mean(
       tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    """
    cross_entropy_mean = tf.reduce_mean(
        tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
            axis=1,
        )
    )

    tf.summary.scalar(
        name_scope + '_cross_entropy',
        cross_entropy_mean
    )
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss))
    return total_loss

def tower_acc(logits, labels):
    correct_pred = tf.equal(tf.round(tf.nn.sigmoid(logits)), tf.round(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def run_model():

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size * N_GPU
        )
        logits = []
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': _variable_with_weight_decay('wout_finetune', [4096, c3d_model.NUM_CLASSES], 0.0005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                'out': _variable_with_weight_decay('bout_finetune', [c3d_model.NUM_CLASSES], 0.000),
            }

        crop_mean = np.load('crop_mean.npy').reshape([c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, 3])
        for i, gpu_index in enumerate(GPU_LIST):
            with tf.device('/gpu:%d' % gpu_index):
                logit, _ = c3d_model.inference_c3d(
                    images_placeholder[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :, :],
                    0.5,
                    FLAGS.batch_size,
                    weights,
                    biases
                )
                logits.append(logit)
        logits = tf.concat(logits, 0)


        # Create a session for running Ops on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True),
        )

        saver = tf.train.Saver()
        saver.restore(sess, MODEL_FPATH)

        n_data = input_data.count_n_data(DATA_FPATH)
        data_start_pos = 0
        t_frames = None
        t_preds = None
        t_actuals = None
        while data_start_pos < n_data:
            print("Processing {}/{}...".format(data_start_pos, n_data))

            """ Log test summary """
            test_clips, test_labels, data_start_pos, test_metadata = input_data.read_clip_and_label(
                metadata_fpath=DATA_FPATH,
                batch_size=FLAGS.batch_size * N_GPU,
                start_pos=data_start_pos,
                num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                crop_size=c3d_model.CROP_SIZE,
                shuffle=False,
                use_person_bbox=False,
            )
            preds = sess.run(
                logits,
                feed_dict={
                    images_placeholder: test_clips,
                    labels_placeholder: test_labels,
                }
            )
            if t_preds is None:
                t_frame_fpaths = [ "{}/{}".format(dpath, fname) for dpath, fname, _ in test_metadata ]
                t_preds = preds
                t_actuals = test_labels
            else:
                t_frame_fpaths += [ "{}/{}".format(dpath, fname) for dpath, fname, _ in test_metadata ]
                t_preds = np.vstack([t_preds, preds])
                t_actuals = np.vstack([t_actuals, test_labels])
            print("len(t_frame_fpaths): ", len(t_frame_fpaths))

        generate_demo_video(t_frame_fpaths, t_preds, t_actuals, show_person_bbox=SHOW_PERSON_BBOX)

    print("done")


def generate_demo_video(frame_fpaths, preds, actuals, show_person_bbox=False):
    with open("./data/index_action.json", "r") as fin:
        index_action_dict = json.load(fin)

    frame = cv2.imread(frame_fpaths[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vout = cv2.VideoWriter("demo.mp4", apiPreference=0, fourcc=fourcc, fps=5, frameSize=(width, height))
    for fpath, pred, actual in zip(frame_fpaths, preds, actuals):
        frame = cv2.imread(fpath)

        actual_label_indices = np.where(actual == 1)[0]
        actual_labels = [index_action_dict[str(int(index))] for index in actual_label_indices]

        pred = 1 / (1 + np.exp(-pred))
        pred_label_indices = np.where(pred > 0.5)[0]
        pred_labels = [index_action_dict[str(int(index))] for index in pred_label_indices]

        is_correct = np.all((actual == 1) == (pred > 0.5))

        # PRED & ACTUAL
        TEXT_HEIGHT = 200
        cv2.putText(
            frame,
            text="Ground Truth: {}".format(", ".join(actual_labels)),
            org=(0, (TEXT_HEIGHT - 5) // 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 0, 0),
            thickness=4,
        )
        cv2.putText(
            frame,
            text="Prediction: {}".format(", ".join(pred_labels)),
            org=(0, TEXT_HEIGHT - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 255),
            thickness=4,
        )

        # BOUNDING BOX
        if show_person_bbox:
            frame_number_parser = parse.compile("{frame_number:d}.jpg")
            _, _, episode_name, frame_fname = fpath.split("/")
            frame_number = frame_number_parser.parse(frame_fname)["frame_number"]
            bbox_fpath = "{}/{}/{:05d}.json".format(BBOX_DPATH, episode_name, frame_number)
            with open(bbox_fpath, "r") as fin:
                bboxes = json.load(fin)
            for bbox in bboxes:
                if bbox["label"] != "person": continue
                x1 = bbox["topleft"]["x"]
                y1 = bbox["topleft"]["y"]
                x2 = bbox["bottomright"]["x"]
                y2 = bbox["bottomright"]["y"]

                cv2.rectangle(
                    frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 255, 255),
                    thickness=4,
                )

        vout.write(frame)
    vout.release()

def main(_):
    run_model()

if __name__ == '__main__':
    tf.app.run()

