
"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import print_function

import json
import os
import re
from collections import defaultdict

import parse

FPS = 5 # five frame per second

def extract_actions_per_frame(episode):
    # load meta data
    meta_fpath = "data/VTT3_meta_data/EP{0:02d}/s01_ep{0:02d}_tag2_visual_Final_180809.json".format(episode)
    with open(meta_fpath, 'rb') as fin:
        """
        for line in fin.readlines():
            line = line.replace(b'\x99', b"'")
            line = line.replace(b'\xbf\xe3\x81\x88', b"")
            try:
                line.decode()
            except:
                print(line)
        """
        data = fin.read()
        data = data.replace(b'\x99', b"'")
        data = data.replace(b'\xbf\xe3\x81\x88', b"")
        data = data.decode('utf-8')


    data = re.sub(r"\s", "", data)
    jdata = json.loads(data)

    num_frames = len(jdata['visual_results'])
    time_parser = parse.compile("{hour:d}:{minute:d}:{second:d};{frame:d}")
    frame_action_dict = {}
    for frame in range(0, num_frames, 3 * 8): # 3 for removing redundant action labeling & 8 from C3D paper
        start_time_string = jdata['visual_results'][frame]['start_time']
        s = time_parser.parse(start_time_string)
        s_time_frm = int((s["hour"] * 3600 + s["minute"] * 60 + s["second"] + s["frame"]/24.0) * FPS)

        end_time_string = jdata['visual_results'][frame]['end_time']
        e = time_parser.parse(end_time_string)
        e_time_frm = int((e["hour"] * 3600 + e["minute"] * 60 + e["second"] + e["frame"]/24.0) * FPS)

        behaviors = []
        names = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']
        for name in names:
            behavior = jdata['visual_results'][frame]['person'][0][name][0]['behavior']
            behaviors.append(behavior)

        # print("[{}] {} ~ {}".format(frame, s_time_frm, e_time_frm))
        for v_frame in range(s_time_frm, e_time_frm):
            actions = list(set(behaviors))
            frame_action_dict[v_frame] = actions
            # print("[{}] {}".format(v_frame, " ".join(actions)))
    return frame_action_dict


def save_frame_action():
    out_dpath = "data/actions"
    os.makedirs(out_dpath, exist_ok=True)
    for episode in range(1, 11):
        frame_action_dict = extract_actions_per_frame(episode)
        with open("{}/S01_E{:02d}.json".format(out_dpath, episode), 'w') as fout:
            json.dump(frame_action_dict, fout)

def plot_action_frame():
    out_dpath = "data/actions"

    from collections import Counter
    from pprint import pprint
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    total_actions = []
    for episode in range(1, 11):
        with open("{}/S01_E{:02d}.json".format(out_dpath, episode), 'r') as fin:
            frame_action_dict = json.load(fin)
            action_lists = list(frame_action_dict.values())
            actions = [action.lower() for action_list in action_lists for action in action_list]
            total_actions += actions
    c = Counter(total_actions)

    action_labels = list(c.keys())
    action_ids = list(range(len(action_labels)))
    action_counts = list(c.values())

    plt.rcParams["figure.figsize"] = [30, 16]
    plt.bar(action_labels, action_counts)
    plt.xticks(rotation=90)

    plt.title("The number of data for each action label")
    plt.xlabel("Action")
    plt.ylabel("Number")
    plt.savefig('action_labels.png')


    from pandas import DataFrame
    sC = { k:c[k] for k in sorted(c.keys()) }
    df = DataFrame(list(sC.values()), index=list(sC.keys()))
    df.to_csv("friends_action.csv")
    pprint(sC)


def list_video_action():
    with open("data/action_index.json", "r") as fin:
        action_index_dict = json.load(fin)

    def filter_actions(actions):
        if len(actions) == 0:
            actions = [ "none" ]

        actions = [ action.lower() for action in actions if len(action) > 0 ]
        action_indices = []
        for action in actions:
            action_indices.append(action_index_dict[action])
        action_indices = list(set(action_indices))

        if "15" in action_indices and len(action_indices) > 1:
            action_indices.remove("15")
        return action_indices

    import os
    import numpy as np

    # N_MAX_DATA_PER_ACTION = float("inf")
    N_MAX_DATA_PER_ACTION = 1000
    TRAIN_RATIO = 0.7
    episodes = [i for i in range(1, 11)]
    n_train_episodes = int(len(episodes) * TRAIN_RATIO)
    train_episodes = episodes[:n_train_episodes]
    test_episodes = episodes[n_train_episodes:]


    """ Group frame-action pairs into train & test set """
    total_actions = []
    video_dnames = os.listdir("data/friends")

    train_video_action_dict = defaultdict(lambda: [])
    for episode in train_episodes:
        episode_video_action_list = []
        with open("data/actions/S01_E{:02d}.json".format(episode), 'r') as fin:
            frame_action_dict = json.load(fin)
        frame_action_list = [ (frame, frame_action_dict[frame]) for frame in sorted(frame_action_dict, key=lambda frame: int(frame)) ]
        video_dname = next((dname for dname in video_dnames if dname.startswith("1x{:02d}".format(episode))), None)
        video_dpath = "data/friends/{}".format(video_dname)
        for frame, actions in frame_action_list:
            # Filter actions
            actions = filter_actions(actions)
            total_actions.append(np.asarray(actions, dtype=int))

            video_fname = "{:05d}.jpg".format(int(frame)+1)
            actions = ",".join([ str(action) for action in actions ])
            video_action_string = "{}\t{}\t{}".format(video_dpath, video_fname, actions)
            train_video_action_dict[actions].append(video_action_string)
            episode_video_action_list.append(video_action_string)
        with open("list/friends_train_s01_e{:02d}.list".format(episode), "w") as fout:
            episode_video_action_string = "\n".join(episode_video_action_list)
            fout.write(episode_video_action_string)

    test_video_action_dict = defaultdict(lambda: [])
    for episode in test_episodes:
        episode_video_action_list = []
        with open("data/actions/S01_E{:02d}.json".format(episode), 'r') as fin:
            frame_action_dict = json.load(fin)
        frame_action_list = [ (frame, frame_action_dict[frame]) for frame in sorted(frame_action_dict) ]
        video_dname = next((dname for dname in video_dnames if dname.startswith("1x{:02d}".format(episode))), None)
        video_dpath = "data/friends/{}".format(video_dname)
        for frame, actions in frame_action_list:
            # Filter actions
            actions = filter_actions(actions)
            total_actions.append(np.asarray(actions, dtype=int))

            video_fname = "{:05d}.jpg".format(int(frame)+1)
            actions = ",".join([ str(action) for action in actions ])
            video_action_string = "{}\t{}\t{}".format(video_dpath, video_fname, actions)
            test_video_action_dict[actions].append(video_action_string)
            episode_video_action_list.append(video_action_string)
        with open("list/friends_test_s01_e{:02d}.list".format(episode), "w") as fout:
            episode_video_action_string = "\n".join(episode_video_action_list)
            fout.write(episode_video_action_string)


    """ Log statistics of total actions """
    from collections import Counter
    from pprint import pprint
    pprint(Counter([ action for actions in total_actions for action in actions ]))


    """ Filter the number of data in each action category """
    np.random.seed(42)

    train_video_action_list = []
    for action, video_action_strings in train_video_action_dict.items():
        np.random.shuffle(video_action_strings)
        n_data = min(N_MAX_DATA_PER_ACTION, len(video_action_strings))
        train_video_action_list += video_action_strings[:n_data]

    test_video_action_list = []
    for action, video_action_strings in test_video_action_dict.items():
        np.random.shuffle(video_action_strings)
        n_data = min(N_MAX_DATA_PER_ACTION, len(video_action_strings))
        test_video_action_list += video_action_strings[:n_data]

    """ Save frame-action pairs """
    np.random.shuffle(train_video_action_list)
    with open("list/friends_train.list".format(N_MAX_DATA_PER_ACTION), "w") as fout:
        train_video_actions = "\n".join(train_video_action_list)
        fout.write(train_video_actions)

    np.random.shuffle(test_video_action_list)
    with open("list/friends_test.list".format(N_MAX_DATA_PER_ACTION), "w") as fout:
        test_video_actions = "\n".join(test_video_action_list)
        fout.write(test_video_actions)


def main():
    save_frame_action()
    list_video_action()

if __name__ == '__main__':
    main()

