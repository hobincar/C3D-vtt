
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
    for frame in range(0, num_frames, 3):
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
    def filter_actions(actions):
        if len(actions) == 0:
            actions = [ "none" ]
        replace = lambda l, k, c: [k if e.lower() in c else e for e in l]

        actions = [ action for action in actions if len(action) > 0 ]

        """
        actions = replace(actions, "call", ["call", "phonecall"])
        actions = replace(actions, "cleanup", ["cleaningup"])
        actions = replace(actions, "cook", ["cooking"])
        actions = replace(actions, "cut", ["cutting"])
        actions = replace(actions, "destroy", ["destroysomething"])
        actions = replace(actions, "drink", ["drinking"])
        actions = replace(actions, "eat", ["eating"])
        actions = replace(actions, "hold", ["cup", "hodlingabottle", "holdingabottle", "holdingacup", "holdingajacket", "holdingapaper", "holdingatelephone", "holdingnewspaper", "holdingpaper", "holdingsomething", "holdsomething"])
        actions = replace(actions, "dance", ["dance", "dancing", "danicng"])
        actions = replace(actions, "find", ["findingsomething", "findsomething"])
        actions = replace(actions, "highfive", ["high-five", "highfive"])
        actions = replace(actions, "hug", ["hugging"])
        actions = replace(actions, "kiss", ["kissing"])
        actions = replace(actions, "lookbackat", ["lookbackat"])
        actions = replace(actions, "nod", ["nodding"])
        actions = replace(actions, "None", ["none", "nonoe"])
        actions = replace(actions, "opendoor", ["openingdoor"])
        actions = replace(actions, "playguitar", ["playingguitar"])
        actions = replace(actions, "pointout", ["pointingout"])
        actions = replace(actions, "pushaway", ["pushingaway", "pusingaway"])
        actions = replace(actions, "putarmsaroundshoulder", ["puttingarmsaroundeachother'sshoulder", "puttingarmsaroundeachother?'sshoulder"])
        actions = replace(actions, "shakehands", ["shakinghands"])
        actions = replace(actions, "sing", ["singing"])
        actions = replace(actions, "sit", ["sittingdown", "sittingon"])
        actions = replace(actions, "smoke", ["smoking"])
        actions = replace(actions, "standup", ["standing", "standingup"])
        actions = replace(actions, "walk", ["walking"])
        actions = replace(actions, "watch", ["watching", "watchingtv"])
        actions = replace(actions, "wavehands", ["wavinghands"])
        actions = replace(actions, "wearlipstick", ["wearinglipstick"])
        """

        actions = replace(actions, "0", ["call", "phonecall"])
        actions = replace(actions, "1", ["cleaningup"])
        actions = replace(actions, "2", ["cooking"])
        actions = replace(actions, "3", ["cutting"])
        actions = replace(actions, "4", ["destroysomething"])
        actions = replace(actions, "5", ["drinking"])
        actions = replace(actions, "6", ["eating"])
        actions = replace(actions, "7", ["cup", "hodlingabottle", "holdingabottle", "holdingacup", "holdingajacket", "holdingapaper", "holdingatelephone", "holdingnewspaper", "holdingpaper", "holdingsomething", "holdsomething"])
        actions = replace(actions, "8", ["dance", "dancing", "danicng"])
        actions = replace(actions, "9", ["findingsomething", "findsomething"])
        actions = replace(actions, "10", ["high-five", "highfive"])
        actions = replace(actions, "11", ["hugging"])
        actions = replace(actions, "12", ["kissing"])
        actions = replace(actions, "13", ["lookbackat"])
        actions = replace(actions, "14", ["nodding"])
        actions = replace(actions, "15", ["none", "nonoe"])
        actions = replace(actions, "16", ["openingdoor"])
        actions = replace(actions, "17", ["playingguitar"])
        actions = replace(actions, "18", ["pointingout"])
        actions = replace(actions, "19", ["pushingaway", "pusingaway"])
        actions = replace(actions, "20", ["puttingarmsaroundeachother'sshoulder", "puttingarmsaroundeachother?'sshoulder"])
        actions = replace(actions, "21", ["shakinghands"])
        actions = replace(actions, "22", ["singing"])
        actions = replace(actions, "23", ["sittingdown", "sittingon"])
        actions = replace(actions, "24", ["smoking"])
        actions = replace(actions, "25", ["standing", "standingup"])
        actions = replace(actions, "26", ["walking"])
        actions = replace(actions, "27", ["watching", "watchingtv"])
        actions = replace(actions, "28", ["wavinghands"])
        actions = replace(actions, "29", ["wearinglipstick"])

        actions = list(set(actions))

        if "15" in actions and len(actions) > 1:
            actions.remove("15")
        return actions

    import os
    import numpy as np

    video_dnames = [ dname for dname in os.listdir("data/friends") if dname.startswith("1x") ]
    video_action_dict = defaultdict(lambda: [])
    total_actions = []
    for episode in range(1, 11):
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
            actions = ",".join(actions)
            video_action_string = "{}\t{}\t{}".format(video_dpath, video_fname, actions)
            video_action_dict[actions].append(video_action_string)

    """
    video_action_list = np.asarray(video_action_list)
    indices = np.arange(0, len(video_action_list))
    np.random.seed(42)
    np.random.shuffle(indices)
    n_train = int(len(video_action_list) * 0.8)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_video_action_list = video_action_list[train_indices]
    train_video_action_list = "\n".join(train_video_action_list)
    with open("list/friends_train.list", "w") as fout:
        fout.write(train_video_action_list)

    val_video_action_list = video_action_list[val_indices]
    val_video_action_list = "\n".join(val_video_action_list)
    with open("list/friends_val.list", "w") as fout:
        fout.write(val_video_action_list)
    """

    # N_MAX_DATA_PER_ACTION = float("inf")
    N_MAX_DATA_PER_ACTION = 1000
    TRAIN_RATIO = 0.7
    np.random.seed(42)
    train_video_action_list = []
    val_video_action_list = []
    for action, video_action_strings in video_action_dict.items():
        np.random.shuffle(video_action_strings)
        n_data = min(N_MAX_DATA_PER_ACTION, len(video_action_strings))
        n_train = int( n_data * TRAIN_RATIO )
        train_video_action_list += video_action_strings[:n_train]
        val_video_action_list += video_action_strings[n_train:n_data]
    np.random.shuffle(train_video_action_list)
    with open("list/friends_train_balanced-{}.list".format(N_MAX_DATA_PER_ACTION), "w") as fout:
        train_video_actions = "\n".join(train_video_action_list)
        fout.write(train_video_actions)
    np.random.shuffle(val_video_action_list)
    with open("list/friends_test_balanced-{}.list".format(N_MAX_DATA_PER_ACTION), "w") as fout:
        val_video_actions = "\n".join(val_video_action_list)
        fout.write(val_video_actions)

    """
    from collections import Counter
    from pprint import pprint
    pprint(Counter([ action for actions in total_actions for action in actions ]))
    """


def main():
    # save_frame_action()
    list_video_action()

if __name__ == '__main__':
    main()

