import json


def parse_action_info(action_info):
    index_actions_dict = {}
    index_action_dict = {}
    action_index_dict = {}

    for index, action, actions in action_info:
        index_actions_dict[index] = actions

        index_action_dict[index] = action

        assert action not in action_index_dict
        action_index_dict[action] = index
        for a in actions:
            assert a == action or a not in action_index_dict
            action_index_dict[a] = index

    return index_actions_dict, index_action_dict ,action_index_dict


if __name__ == "__main__":
    index_actions_dict, index_action_dict, action_index_dict = parse_action_info([
        [ 0, "call", [ "call", "phonecall" ] ],
        [ 1, "cleanup", [ "cleaningup" ] ],
        [ 2, "cook", [ "cooking" ] ],
        [ 3, "cut", [ "cutting" ] ],
        [ 4, "destroy", [ "destroysomething" ] ],
        [ 5, "drink", [ "drinking" ] ],
        [ 6, "eat", [ "eating" ] ],
        [ 7, "hold", [ "cup", "hodlingabottle", "holdingabottle", "holdingacup", "holdingajacket", "holdingapaper", "holdingatelephone", "holdingnewspaper", "holdingpaper", "holdingsomething", "holdsomething" ] ],
        [ 8, "dance", [ "dance", "dancing", "danicng" ] ],
        [ 9, "find", [ "findingsomething", "findsomething" ] ],
        [ 10, "highfive", [ "high-five", "highfive" ] ],
        [ 11, "hug", [ "hugging" ] ],
        [ 12, "kiss", [ "kissing" ] ],
        [ 13, "lookbackat", [ "lookbackat" ] ],
        [ 14, "nod", [ "nodding" ] ],
        [ 15, "None", [ "none", "nonoe" ] ],
        [ 16, "opendoor", [ "openingdoor" ] ],
        [ 17, "playguitar", [ "playingguitar" ] ],
        [ 18, "pointout", [ "pointingout" ] ],
        [ 19, "pushaway", [ "pushingaway", "pusingaway" ] ],
        [ 20, "putarmsaroundshoulder", [ "puttingarmsaroundeachother'sshoulder", "puttingarmsaroundeachother?'sshoulder" ] ],
        [ 21, "shakehands", [ "shakinghands" ] ],
        [ 22, "sing", [ "singing" ] ],
        [ 23, "sit", [ "sittingdown", "sittingon" ] ],
        [ 24, "smoke", [ "smoking" ] ],
        [ 25, "standup", [ "standing", "standingup" ] ],
        [ 26, "walk", [ "walking" ] ],
        [ 27, "watch", [ "watching", "watchingtv" ] ],
        [ 28, "wavehands", [ "wavinghands" ] ],
        [ 29, "wearlipstick", [ "wearinglipstick" ] ],
    ])

    with open("./data/index_actions.json", "w") as fout:
        json.dump(index_actions_dict, fout)
    with open("./data/index_action.json", "w") as fout:
        json.dump(index_action_dict, fout)
    with open("./data/action_index.json", "w") as fout:
        json.dump(action_index_dict, fout)

