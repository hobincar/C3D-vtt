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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import time

import tensorflow as tf
import PIL.Image as Image
import numpy as np
import cv2
import parse
from tqdm import tqdm

import c3d_model


def read_frame(frame_fpath, crop_size, crop_mean):
    frame = Image.open(frame_fpath)
    np_frame = np.array(frame)

    if frame.width > frame.height:
        scale = crop_size / frame.height
        img = np.array(cv2.resize(np_frame, (int(frame.width * scale + 1), crop_size))).astype(np.float32)
    else:
        scale = crop_size / frame.width
        img = np.array(cv2.resize(np_frame, (crop_size, int(img.height * scale + 1)))).astype(np.float32)

    img = img[np.int((img.shape[0] - crop_size)/2):np.int((img.shape[0] - crop_size)/2) + crop_size,
              np.int((img.shape[1] - crop_size)/2):np.int((img.shape[1] - crop_size)/2) + crop_size,:] - crop_mean

    return img

def read_frame_with_bbox_tight(frame_fpath, crop_size, crop_mean, bbox):
    frame = Image.open(frame_fpath)
    np_frame = np.array(frame)

    np_frame = np_frame[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax'], :]

    img = np.array(cv2.resize(np_frame, (crop_size, crop_size))).astype(np.float32) # - crop_mean

    return img


def read_frame_with_bbox_loose(frame_fpath, crop_size, crop_mean, bbox):
    frame = Image.open(frame_fpath)
    np_frame = np.array(frame)

    bbox_width = bbox['xmax'] - bbox['xmin']
    bbox_height = bbox['ymax'] - bbox['ymin']
    if bbox_width > bbox_height:
        xmin = bbox['xmin']
        xmax = bbox['xmax']
        ymin = int((bbox['ymin'] + bbox['ymax']) / 2 - bbox_width / 2)
        ymax = ymin + bbox_width
    else:
        xmin = int((bbox['xmin'] + bbox['xmax']) / 2 - bbox_height / 2)
        xmax = xmin + bbox_height
        ymin = bbox['ymin']
        ymax = bbox['ymax']
    img = np_frame[ymin:ymax, xmin:xmax, :]

    img = np.array(cv2.resize(img, (crop_size, crop_size))).astype(np.float32)

    return img

def read_clip_and_label(metadata_fpath, batch_size, start_pos=-1, num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP, crop_size=c3d_model.CROP_SIZE, shuffle=False, use_cached=False, use_person_bbox=False):
    with open(metadata_fpath, 'r') as fin:
        rows = list(fin)
        start_pos = start_pos % len(rows)
        if start_pos == -1 or shuffle:
            np.random.seed(seed=42)
            np.random.shuffle(rows)
            rows = rows[:batch_size]
        else:
            rows = rows[start_pos:start_pos+batch_size]
        metadata = [ row.strip("\n").split() for row in rows ]
    next_start_pos = start_pos + len(metadata)


    crop_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    frame_number_parser = parse.compile("{frame_number:d}.jpg")
    clips = []
    labels = []
    for frame_dpath, frame_fname, action_label in metadata:
        start_frame_number = frame_number_parser.parse(frame_fname)["frame_number"]
        max_frame_number = max([ frame_number_parser.parse(fname)["frame_number"] for fname in os.listdir(frame_dpath) if fname.endswith(".jpg") ])
        if start_frame_number + num_frames_per_clip - 1 > max_frame_number:
            continue

        cached_clip_fpath = "{}/{:05d}.clip{}.npy".format(
            frame_dpath,
            start_frame_number,
            ".person_bbox" if use_person_bbox else ""
        )
        if use_cached and os.path.isfile(cached_clip_fpath):
            with open(cached_clip_fpath, 'r') as fin:
                clip = np.load(cached_clip_fpath)
        else:
            clip = []
            for i, frame_number in enumerate(range(start_frame_number, start_frame_number + num_frames_per_clip)):
                frame_fpath = "{}/{:05d}.jpg".format(frame_dpath, frame_number)
                bbox = None
                if use_person_bbox:
                    episode_name = frame_dpath.split("/")[-1]
                    person_bbox_fpath = "./data/friends_json/bbox/person/{}/{:05d}.json".format(episode_name, frame_number)
                    with open(person_bbox_fpath, 'r') as fin:
                        person_bboxes = json.load(fin)
                    min_xmin = float("inf")
                    min_ymin = float("inf")
                    max_xmax = -float("inf")
                    max_ymax = -float("inf")
                    for person_bbox in person_bboxes:
                        if person_bbox['label'] != 'person': continue
                        min_xmin = min(min_xmin, int(person_bbox['topleft']['x']))
                        min_ymin = min(min_ymin, int(person_bbox['topleft']['y']))
                        max_xmax = max(max_xmax, int(person_bbox['bottomright']['x']))
                        max_ymax = max(max_ymax, int(person_bbox['bottomright']['y']))
                    bbox = {
                        'xmin': min_xmin,
                        'ymin': min_ymin,
                        'xmax': max_xmax,
                        'ymax': max_ymax,
                    }
                    if np.any(np.isinf([ min_xmin, min_ymin, max_xmax, max_ymax ])):
                        np_frame = read_frame(frame_fpath, crop_size, crop_mean[i])
                    else:
                        np_frame = read_frame_with_bbox_tight(frame_fpath, crop_size, crop_mean[i], bbox)
                else:
                    np_frame = read_frame(frame_fpath, crop_size, crop_mean[i])
                clip.append(np_frame)
            np.save(cached_clip_fpath, clip)
        clips.append(clip)
        action_label = action_label.split(",")
        action_label = [ int(l) for l in action_label ]
        onehot_action_label = np.zeros(c3d_model.NUM_CLASSES, dtype=np.float32)
        onehot_action_label[action_label] = 1
        labels.append(onehot_action_label)

    # pad (duplicate) data/label if less than batch_size
    assert len(clips) == len(labels)
    if len(clips) < batch_size:
        for _ in range(batch_size - len(clips)):
            clips.append(clip)
            labels.append(onehot_action_label)

    np_clips = np.array(clips).astype(np.float32)
    np_labels = np.array(labels).astype(np.float32)
    return np_clips, np_labels, next_start_pos, metadata


def count_n_data(data_fpath):
    with open(data_fpath, "r") as fin:
        n_data = len(fin.readlines())
    return n_data

