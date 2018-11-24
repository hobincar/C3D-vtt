"""
Format:
    S01_EP01    13  17    4, 7, 8, 9
    ...
"""

import json
import os
import random
random.seed(42)

import parse

from config import ListConfig as C

with open("data/act2idx.json", 'r') as fin:
    act2idx = json.load(fin)
with open("data/idx2rep.json", 'r') as fin:
    idx2rep = json.load(fin)
with open("data/rep2sta.json", 'r') as fin:
    rep2sta = json.load(fin)
action_labels = list(set(act2idx.values()))
FPS_USED_TO_EXTRACT_FRAMES = 5.07


def load_annotations():
    annotations = []
    root_dpath = "data/friends_trimmed/annotations"
    for fname in os.listdir(root_dpath):
        fpath = os.path.join(root_dpath, fname)
        with open(fpath, 'r') as fin:
            annotation = json.load(fin)
        annotations.append(annotation)
    return annotations


def parse_episode(fname):
    p = parse.compile("{}.json")
    episode = p.parse(fname)[0]
    return episode


def parse_frame_number(fname):
    p = parse.compile("{:d}.jpg")
    frame_number = p.parse(fname)[0]
    return frame_number


def timestr_to_seconds(timestr):
    time_parser = parse.compile("{:d}:{:d}:{:d};{:d}")
    h, m, s, ms = time_parser.parse(timestr)
    seconds = 3600*h + 60*m + s + 1/60*ms
    return seconds


def parse_annotation(annotation):
    start_time = annotation["start_time"]
    start_seconds = timestr_to_seconds(start_time)
    start_frame = int(start_seconds * FPS_USED_TO_EXTRACT_FRAMES)

    end_time = annotation["end_time"]
    end_seconds = timestr_to_seconds(end_time)
    end_frame = int(end_seconds * FPS_USED_TO_EXTRACT_FRAMES)

    labels = []
    for person, info in annotation["person"][0].items():
        info = info[0]

        action = info["behavior"]
        if action in act2idx:
            action_index = act2idx[action]
            labels.append(action_index)
    labels = list(set(labels))
    
    return start_frame, end_frame, labels


def generate_list():
    annotations_list = load_annotations()
    total_list = []
    for annotations in annotations_list:
        episode = parse_episode(annotations["file_name"])

        frame_fnames = os.listdir(os.path.join("data/friends_trimmed/frames", episode))
        frame_numbers = [ parse_frame_number(fname) for fname in frame_fnames ]
        terminal_frame = max(frame_numbers)
        for annotation in annotations["visual_results"]:
            start_frame, end_frame, labels = parse_annotation(annotation)
            if len(labels) == 0: continue

            n_frames = end_frame + 1 - start_frame
            n_front = C.n_frames_per_clip // 2 - 1
            n_back = C.n_frames_per_clip - n_front
            if n_frames < C.n_frames_per_clip:
                median_frame = (start_frame + end_frame) // 2
                start_frame = median_frame - n_front
                end_frame = median_frame + n_back
                if start_frame < 1: continue
                if end_frame > terminal_frame: continue
                total_list.append(( episode, start_frame, end_frame, labels ))
            else:
                for median_frame in range(start_frame+n_front, end_frame-n_back+1, 16):
                    start_frame = median_frame - n_front
                    end_frame = median_frame + n_back
                    if start_frame < 1: continue
                    if end_frame > terminal_frame: continue
                    total_list.append(( episode, start_frame, end_frame, labels ))
    return total_list


def split_list(total_list):
    # Get the indices of total list according to an action index.
    actionLabel_listIdxs_dict = { label: [] for label in action_labels }
    for i, d in enumerate(total_list):
        _, _, _, labels = d
        for label in labels:
            actionLabel_listIdxs_dict[label].append(i)

    # Sort action labels along its n_clips.
    sorted_action_labels = sorted(action_labels, key=lambda l: rep2sta[idx2rep[str(l)]]["n_clips"])

    already_taken = [ False for _ in range(len(total_list)) ]
    train_idxs = []
    test_idxs = []
    for action_label in sorted_action_labels:
        list_idxs = actionLabel_listIdxs_dict[action_label]
        list_idxs = [ idx for idx in list_idxs if not already_taken[idx] ]
        random.shuffle(list_idxs)
        n_train = int(len(list_idxs) * C.train_ratio)
        train = list_idxs[:n_train]
        test = list_idxs[n_train:]

        for i in train:
            train_idxs.append(i)
            already_taken[i] = True
        for i in test:
            test_idxs.append(i)
            already_taken[i] = True
    assert all(already_taken)

    total_list = [ [ ep, str(s), str(e), ','.join([ str(l) for l in ls ]) ] for ep, s, e, ls in total_list ]
    train_list = [ '\t'.join(total_list[i]) for i in train_idxs ]
    test_list = [ '\t'.join(total_list[i]) for i in test_idxs ]
    return train_list, test_list


if __name__ == "__main__":
    total_list = generate_list()
    train_list, test_list = split_list(total_list)

    with open("list/friends_train.list", 'w') as fout:
        fout.write('\n'.join(train_list))
    with open("list/friends_test.list", 'w') as fout:
        fout.write('\n'.join(test_list))

