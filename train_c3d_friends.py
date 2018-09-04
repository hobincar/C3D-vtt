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

import c3d_model
import input_data
from logger import Logger


# Basic model parameters as external flags.
flags = tf.app.flags
GPU_LIST = [0, 1]
N_GPU = len(GPU_LIST)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([ str(i) for i in GPU_LIST ])
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
MODEL_TAG = "friends"
MODEL_SAVE_DPATH = './models/{}'.format(MODEL_TAG)
LOG_DPATH =  "./visual_logs/{}".format(MODEL_TAG)
TRAIN_DATA_FPATH = "./list/friends_train.list"
TEST_DATA_FPATH = "./list/friends_test.list"


with open("./data/index_action.json", "r") as fin:
    index_action_dict = json.load(fin)

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

def calc_metrics(preds, actuals):
    preds = (preds > 0.5).astype(np.bool)
    actuals = actuals.astype(np.bool)

    TP = np.logical_and(preds, actuals).sum(axis=1)
    FP = np.logical_and(preds, ~actuals).sum(axis=1)
    FN = np.logical_and(~preds, actuals).sum(axis=1)

    precision = TP / (TP + FP)
    precision[np.isnan(precision)] = 0
    precision = precision.mean()
    recall = TP / (TP + FN)
    recall[np.isnan(recall)] = 0
    recall = recall.mean()
    f1score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1score

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

def pred_real_to_table(preds, reals, prefix=""):
    TOP_K = 3
    lines = [
        [ "real", "pred" ],
    ]
    for i, (pred, real) in enumerate(zip(preds, reals), 1):
        # n_label = (real == 1).sum()
        real_label_indices = np.where(real == 1)[0]
        real_label_indices = ", ".join([ str(i) for i in real_label_indices ])
        # pred_softmax = (np.e ** pred) / (np.e ** pred).sum()
        pred_sigmoid = 1 / (1 + np.exp(-pred))
        pred_label_indices = np.argsort(-pred)[:TOP_K]
        # pred_label_indices = ", ".join([ "{}({:.2f})".format(i, pred_softmax[i]) for i in pred_label_indices ])
        pred_label_indices = ", ".join([ "{}({:.2f})".format(i, pred_sigmoid[i]) for i in pred_label_indices ])
        lines.append([ real_label_indices, pred_label_indices ])
    return lines

def clip_summary_with_text(clip, actual, pred):
    TEXT_HEIGHT = 25
    TEXT_WIDTH = 50
    padded_clip = np.pad(clip, pad_width=((0,0), (TEXT_HEIGHT,0), (TEXT_WIDTH//2,TEXT_WIDTH//2), (0,0)), mode="constant", constant_values=0)
    actual_labels = np.where(actual == 1)[0]
    pred = (1 / (1 + np.exp(-pred)))
    pred_labels = np.where(pred > 0.5)[0]
    for frame in padded_clip:
        actual_actions = [index_action_dict[str(a)] for a in actual_labels]
        cv2.putText(
            frame,
            "actual: {}".format(", ".join(actual_actions)),
            (0, (TEXT_HEIGHT - 5) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255)
        )
        pred_actions = [index_action_dict[str(p)] for p in pred_labels]
        cv2.putText(
            frame,
            "pred: {}".format(", ".join(pred_actions)),
            (0, TEXT_HEIGHT - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255)
        )
    return padded_clip

def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    os.makedirs(MODEL_SAVE_DPATH, exist_ok=True)
    use_pretrained_model = True
    model_filename = "./sports1m_finetuning_ucf101.model"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size * N_GPU
        )
        tower_grads1 = []
        tower_grads2 = []
        logits = []
        opt_stable = tf.train.AdamOptimizer(1e-4)
        opt_finetuning = tf.train.AdamOptimizer(1e-3)
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
                varlist2 = [ weights['out'], biases['out'] ]
                varlist1 = list( set(list(weights.values()) + list(biases.values())) - set(varlist2) )
                logit, _ = c3d_model.inference_c3d(
                    images_placeholder[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :, :],
                    0.5,
                    FLAGS.batch_size,
                    weights,
                    biases
                )
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(
                    loss_name_scope,
                    logit,
                    labels_placeholder[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                )

                grads1 = opt_stable.compute_gradients(loss, varlist1)
                grads2 = opt_finetuning.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                logits.append(logit)

        logits = tf.concat(logits, 0)
        accuracy = tower_acc(logits, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)
        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt_stable.apply_gradients(grads1)
        apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(list(weights.values()) + list(biases.values()))
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True),
        )
        sess.run(init)
        if os.path.isfile(model_filename) and use_pretrained_model:
            variables = list(weights.values()) + list(biases.values())
            variables_to_restore = [v for v in variables if "finetune" not in v.name]
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_filename)

        # Create summary writter
        merged = tf.summary.merge_all()
        timestamp = datetime.now().strftime('%m/%d_%H:%M')
        train_logger = Logger("{}/train/{}".format(LOG_DPATH, timestamp))
        test_logger = Logger("{}/test/{}".format(LOG_DPATH, timestamp))

        train_start_pos = 0
        test_start_pos = 0
        for step in range(FLAGS.max_steps):
            train_clips, train_labels, train_start_pos, train_metadata = input_data.read_clip_and_label(
                metadata_fpath=TRAIN_DATA_FPATH,
                batch_size=FLAGS.batch_size * N_GPU,
                start_pos=train_start_pos,
                num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                crop_size=c3d_model.CROP_SIZE,
                shuffle=False,
                use_cached=True,
            )
            sess.run(train_op, feed_dict={
                images_placeholder: train_clips,
                labels_placeholder: train_labels
            })
            if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
                """ Save trained model """
                saver.save(sess, os.path.join(MODEL_SAVE_DPATH, 'c3d_friends_model'), global_step=step)


                """ Log train summary """
                summary, preds, acc = sess.run(
                    [merged, logits, accuracy],
                    feed_dict={
                        images_placeholder: train_clips,
                        labels_placeholder: train_labels,
                    }
                )
                print("Train acc.: {:.5f}".format(acc), end="")
                pred_summary = pred_real_to_table(preds, train_labels)
                gif_summary = clip_summary_with_text(train_clips[0] + crop_mean, train_labels[0], preds[0])
                train_logger.scalar_summary("accuracy", acc, step)
                train_logger.text_summary("prediction", pred_summary, step)
                train_logger.gif_summary("clip", gif_summary, step)


                """ Log test summary """
                t_preds = None
                t_actuals = None
                for _ in range(10):
                    test_clips, test_labels, test_start_pos, test_metadata = input_data.read_clip_and_label(
                        metadata_fpath=TEST_DATA_FPATH,
                        batch_size=FLAGS.batch_size * N_GPU,
                        start_pos=test_start_pos,
                        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                        crop_size=c3d_model.CROP_SIZE,
                        shuffle=False,
                        use_cached=True,
                    )
                    summary, preds, acc = sess.run(
                        [merged, logits, accuracy],
                        feed_dict={
                            images_placeholder: test_clips,
                            labels_placeholder: test_labels,
                        }
                    )
                    t_preds = preds if t_preds is None else np.vstack([t_preds, preds])
                    t_actuals = test_labels if t_actuals is None else np.vstack([t_actuals, test_labels])
                precision, recall, f1score = calc_metrics(t_preds, t_actuals)
                print("\tTest acc.: {:.5f}".format(acc))
                pred_summary = pred_real_to_table(preds, test_labels)
                gif_summary = clip_summary_with_text(test_clips[0] + crop_mean, test_labels[0], preds[0])
                test_logger.scalar_summary("accuracy", acc, step)
                test_logger.scalar_summary("precision", precision, step)
                test_logger.scalar_summary("recall", recall, step)
                test_logger.scalar_summary("f1score", f1score, step)
                test_logger.text_summary("prediction", pred_summary, step)
                test_logger.gif_summary("clip", gif_summary, step)
    print("done")

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

