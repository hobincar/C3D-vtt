from collections import defaultdict
import json
import os
import time

import numpy as np


def weight_classes(list_fpath, n_actions):
    with open(list_fpath, 'r') as fin:
        data = fin.readlines()

    label_counter = defaultdict(lambda: 0)
    for d in data:
        _, _, _, labels = d.split('\t')
        labels = [ int(l) for l in labels.split(",") ]
        for label in labels:
            label_counter[label] += 1

    weights = [ None for _ in range(n_actions) ]
    for class_idx, n_data in label_counter.items():
        weights[class_idx] = 1 / n_data
    return np.asarray(weights)


class ListConfig:
    fps_used_to_extract_frames = 5.07
    n_frames_per_clip = 16
    train_ratio = 0.7


class TrainConfig:
    model_tag = "C3D"
    n_workers = 4

    """ Dataset """
    train_data_fpath = "list/friends_train.list"
    test_data_fpath = "list/friends_test.list"

    use_pretrained_model = True
    pretrained_model_fpath = "pretrained_models/sports1m_finetuning_ucf101.model"
    crop_mean_fpath = "data/crop_mean.npy"

    n_iterations = 100000
    train_log_every = 10
    test_log_every = 100
    save_every = 10000
    batch_size = 20
    n_frames_per_clip = ListConfig.n_frames_per_clip
    resize_shape = [ 128, 171 ]
    crop_size = 112
    n_channels = 3
    moving_average_decay = 0.9999
    lr_stable = 1e-5
    lr_finetune = 1e-4

    with open('data/idx2rep.json', 'r') as fin:
        n_actions = len(json.load(fin))

    class_weights = weight_classes(train_data_fpath, n_actions) 


    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    id = "{} | {}".format(model_tag, timestamp)

    log_root_dpath = "logs"
    log_dpath = os.path.join(log_root_dpath, id)

    model_root_dpath = "models"
    model_fpath = os.path.join(model_root_dpath, id, "model")

