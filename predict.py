from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import tensorflow as tf
from tqdm import tqdm
import numpy as np

import c3d_model
from config import PredConfig as C
from dataset import load_train_dataset, load_test_dataset


# Basic model parameters
GPU_LIST = [ int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
N_GPU = len(GPU_LIST)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def placeholder_inputs():
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
    images_placeholder = tf.placeholder(tf.float32, shape=(
        N_GPU * C.batch_size,
        C.n_frames_per_clip,
        C.crop_size,
        C.crop_size,
        C.n_channels))
    labels_placeholder = tf.placeholder(tf.float32, shape=(N_GPU * C.batch_size, C.n_actions))
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
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
    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs()
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
            'out': _variable_with_weight_decay('wout_finetune', [4096, C.n_actions], 0.04, 0.005)
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
            'out': _variable_with_weight_decay('bout_finetune', [C.n_actions], 0.04, 0.0),
        }
    logits = []
    for i, gpu_index in enumerate(GPU_LIST):
        with tf.device('/gpu:%d' % gpu_index):
            logit, _ = c3d_model.inference_c3d(
                images_placeholder[i * C.batch_size:(i + 1) * C.batch_size,:,:,:,:],
                1.,
                C.batch_size,
                weights,
                biases)
            logits.append(logit)
    logits = tf.concat(logits, 0)
    norm_scores = tf.nn.sigmoid(logits)

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    os.makedirs(C.prediction_dpath, exist_ok=True)
    pbar = tqdm(total=sum([ len(episodes) for episodes in C.episodes_list ]))
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            pbar.set_description("Generating prediction results of S{:02d}_EP{:02d}...".format(season, episode))

            list_file_fpath = C.list_fpath_tpl.format(season, episode)
            results = []
    
            # Load train dataset
            dataset = load_test_dataset(list_file_fpath, N_GPU * C.batch_size, shuffle=False, repeat=False)
            # dataset = load_train_dataset(list_file_fpath, N_GPU * C.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            sess.run(iterator.initializer)
    
            while True:
                try:
                    clips, labels, frames = sess.run(next_batch)
                    # clips, labels = sess.run(next_batch)
                except tf.errors.OutOfRangeError:
                    break
                frames = frames.tolist()
    
                predict_scores = norm_scores.eval(
                    session=sess,
                    feed_dict={ images_placeholder: clips })
    
                topk_idxs = np.argsort(predict_scores, axis=1)[:, -C.topk:]
                topk_scores = np.take(predict_scores, topk_idxs)
                topk_scores = topk_scores.tolist()
                for frame, labels, topk_idx, topk_score in zip(frames, labels, topk_idxs, topk_scores):
                    ground_truths = np.where(labels == 1)[0]
                    ground_truths = [ C.idx2rep[str(idx)] for idx in ground_truths ]
    
                    result = {}
                    result["frame"] = frame
                    result["seconds"] = frame / C.fps_used_to_extract_frames
                    result["ground_truths"] = ground_truths
                    actions = []
                    for idx, score in zip(topk_idx, topk_score):
                        action = C.idx2rep[str(idx)]
                        actions.append(( action, score ))
                    result["actions"] = actions
    
                    results.append(result)
            episode_id = "S{:02d}_EP{:02d}".format(season, episode)
            result = {
                "file_name": "{}.json".format(episode_id),
                "registed_name": "{}.json".format(episode_id),
                "prediction_results": results,
            }
            result_fpath = C.prediction_fpath_tpl.format(season, episode)
            with open(result_fpath, 'w') as fout:
                json.dump(result, fout, indent=2, sort_keys=True)

            pbar.update(1)
    
    
def main(_):
    run_test()
    
    
if __name__ == '__main__':
    tf.app.run()

