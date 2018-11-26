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


def load_annotation(season, episode):
    annotation_fpath = C.annotation_fpath_tpl.format(season, episode)
    with open(annotation_fpath, 'r') as fin:
        annotation = json.load(fin)
    return annotation


def load_annotations():
    annotations = []
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            annotation = load_annotation(season, episode)
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
    start_frame = int(start_seconds * C.fps_used_to_extract_frames)

    end_time = annotation["end_time"]
    end_seconds = timestr_to_seconds(end_time)
    end_frame = int(end_seconds * C.fps_used_to_extract_frames)

    labels = []
    for person, info in annotation["person"][0].items():
        info = info[0]

        action = info["behavior"]
        if action in C.act2idx:
            action_index = C.act2idx[action]
            labels.append(action_index)
    labels = list(set(labels))
    
    return start_frame, end_frame, labels


def get_endpoints_from_median_frame(median_frame):
    start_frame = median_frame - C.n_front
    end_frame = median_frame + C.n_back
    return start_frame, end_frame


def get_total_list():
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
            if n_frames < C.n_frames_per_clip:
                median_frame = (start_frame + end_frame) // 2
                start_frame, end_frame = get_endpoints_from_median_frame(median_frame)
                if start_frame < 1: continue
                if end_frame > terminal_frame: continue
                total_list.append(( episode, start_frame, end_frame, labels ))
            else:
                for median_frame in range(start_frame + C.n_front, end_frame - C.n_back + 1, 16):
                    start_frame, end_frame = get_endpoints_from_median_frame(median_frame)
                    if start_frame < 1: continue
                    if end_frame > terminal_frame: continue
                    total_list.append(( episode, start_frame, end_frame, labels ))
    return total_list


def split_list(total_list):
    # Get the indices of total list according to an action index.
    actionLabel_listIdxs_dict = { label: [] for label in C.actions }
    for i, d in enumerate(total_list):
        _, _, _, labels = d
        for label in labels:
            actionLabel_listIdxs_dict[label].append(i)

    # Sort action labels along its n_clips.
    sorted_actions = sorted(C.actions, key=lambda l: C.rep2sta[C.idx2rep[str(l)]]["n_clips"])

    already_taken = [ False for _ in range(len(total_list)) ]
    train_idxs = []
    test_idxs = []
    for action in sorted_actions:
        list_idxs = actionLabel_listIdxs_dict[action]
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


def get_episode_list(season, episode):
    annotations = load_annotation(season, episode)

    frame_fnames = os.listdir(os.path.join("data/friends_trimmed/frames", "S{:02d}_EP{:02d}".format(season, episode)))
    frame_numbers = [ parse_frame_number(fname) for fname in frame_fnames ]
    terminal_frame = max(frame_numbers)

    episode_list = []
    for annotation in annotations["visual_results"]:
        start_frame, end_frame, labels = parse_annotation(annotation)
        for median_frame in range(start_frame, end_frame):
            start, end = get_endpoints_from_median_frame(median_frame)
            if start < 1: continue
            if end > terminal_frame: continue
            episode_list.append((
                "S{:02d}_EP{:02d}".format(season, episode),
                str(start),
                str(end),
                ','.join([ str(label) for label in labels ]) ))

    remove_dups = {}
    for ep, s, e, ls in episode_list:
        remove_dups[s] = ( ep, s, e, ls )
    episode_list = list(remove_dups.values())
    episode_list = sorted(episode_list, key=lambda l: int(l[1]))
    episode_list = [ '\t'.join(l) for l in episode_list ]
    return episode_list


if __name__ == "__main__":
    # Train & Test
    total_list = get_total_list()
    train_list, test_list = split_list(total_list)

    with open("list/friends_train.list", 'w') as fout:
        fout.write('\n'.join(train_list))
    with open("list/friends_test.list", 'w') as fout:
        fout.write('\n'.join(test_list))

    # Episodes
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            episode_list = get_episode_list(season, episode)

            episode_list_fpath = C.list_fpath_tpl.format(season, episode)
            with open(episode_list_fpath, 'w') as fout:
                fout.write('\n'.join(episode_list))

