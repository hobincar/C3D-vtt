import json
import os

import cv2
from tqdm import tqdm

from config import DemoConfig as C


def generate_frame(season, episode, frame_number, ground_truths, actions, height, width, layers):
    frame_fpath = C.frame_fpath_tpl.format(season, episode, frame_number)
    frame = cv2.imread(frame_fpath)

    predictions = [ action for action, score in actions ]

    # PRED & ACTUAL
    TEXT_HEIGHT = 200
    cv2.putText(
        frame,
        text="Ground Truth: {}".format(", ".join(ground_truths)),
        org=(0, (TEXT_HEIGHT - 5) // 2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(255, 0, 0),
        thickness=4,
    )
    cv2.putText(
        frame,
        text="Prediction: {}".format(", ".join(predictions)),
        org=(0, TEXT_HEIGHT - 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 0, 255),
        thickness=4,
    )

    return frame



def generate_demo(season, episode):
    episode_id = "S{:02d}_EP{:02d}".format(season, episode)
    prediction_fpath = C.prediction_fpath_tpl.format(season, episode)
    with open(prediction_fpath, 'r') as fin:
        prediction = json.load(fin)
        prediction_results = prediction["prediction_results"]

    tmp_frame_number =  prediction_results[0]["frame"]
    tmp_frame_fpath = C.frame_fpath_tpl.format(season, episode, tmp_frame_number)
    tmp_frame = cv2.imread(tmp_frame_fpath)
    height, width, layers = tmp_frame.shape

    demo_fpath = C.demo_fpath_tpl.format(season, episode)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vout = cv2.VideoWriter(demo_fpath, apiPreference=0, fourcc=fourcc, fps=5, frameSize=(width, height))
    for pred in prediction_results:
        frame_number = pred["frame"]
        ground_truths = pred["ground_truths"]
        actions = pred["actions"]
        
        frame = generate_frame(season, episode, frame_number, ground_truths, actions, height, width, layers)
        vout.write(frame)
    vout.release()


def generate_demos():
    pbar = tqdm(total=sum([ len(episodes) for episodes in C.episodes_list ]))
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            pbar.set_description("Generating a demo video for S{:02}_EP{:02d}".format(season, episode))

            os.makedirs(C.demo_dpath, exist_ok=True)
            generate_demo(season, episode)

            pbar.update(1)


if __name__ == '__main__':
    generate_demos()

