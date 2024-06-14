import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# TODO: Make paths agnostic to OS.
def move_and_replace(src: Path, dst: Path):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(src, dst)


def getBoxesCoords(boxes):
    xCoords = [box[0] + box[2] / 2 for box in boxes]
    yCoords = [box[1] + box[3] / 2 for box in boxes]
    return xCoords, yCoords


def parseJSONLabels(path):
    labelDict = json.load(open(path, "r"))
    exist, boxes = labelDict.values()
    exist, boxes = np.array(exist, dtype=np.object_), np.array(boxes, dtype=np.object_)

    frames = np.linspace(0, len(exist) - 1, len(exist))  # Videos are 0-indexed
    frames, boxes = frames[exist == 1], boxes[exist == 1]
    return frames, boxes


def parseCSV(path):
    df = pd.read_csv(path)
    frames = df["frame"]
    xCoords = df["x"] + df["w"] / 2
    yCoords = df["y"] + df["h"] / 2
    return np.array([xCoords, yCoords, frames]).T


def COCOToYOLOBox(box, imgXY):
    """Convert from x1,y1,w,h to normalised xc,yc,w,h"""
    x, y, w, h = box
    imgX, imgY = imgXY
    return [(x + w / 2) / imgX, (y + h / 2) / imgY, w / imgX, h / imgY]


def YOLOToCOCOBox(box, imgXY):
    """Convert from normalised xc,yc,w,h to x1,y1,w,h"""
    xc, yc, w, h = box
    imgX, imgY = imgXY
    return [(xc - w / 2) * imgX, (yc - h / 2) * imgY, w * imgX, h * imgY]


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def getLabelPaths(imagePaths: np.ndarray):
    labelPaths = np.char.replace(imagePaths.astype(str), "/images/", "/labels/")
    labelPaths = np.char.replace(labelPaths, ".jpg", ".txt")
    labelPaths = np.char.replace(labelPaths, ".png", ".txt")
    return labelPaths


def truncate_video(input_video_path, output_video_path, start_time, end_time):
    cap = cv2.VideoCapture(str(input_video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    out = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # cap.set may fail for videos with B-frames
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < start_frame:
        _, frame = cap.read()

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        success, frame = cap.read()
        if success:
            out.write(frame)
        else:
            break

    out.release()
    cap.release()
