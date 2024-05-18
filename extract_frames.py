import csv
import os
import shutil

import cv2

CSV_FILE_PATH = "frames.csv"
VIDEO_FILE_PATH = ""
OUTPUT_VIDEO_NAME = "output"
FRAME_NUM = 1
VIDEO_FOLDER = ""

FOLDERS: list[str] = []


def initialiseVar(num: int):
    global VIDEO_FILE_PATH, FRAME_NUM, VIDEO_FOLDER
    VIDEO_FILE_PATH = str(num) + "_front_yolov8n_210324_1_output.avi"
    FRAME_NUM = 1
    VIDEO_FOLDER = VIDEO_FILE_PATH[:-4]
    os.makedirs(VIDEO_FOLDER)
    FOLDERS.append(VIDEO_FOLDER)


with open(CSV_FILE_PATH, mode="r") as file:
    csv_reader = csv.DictReader(file)
    FIRST = True
    prev = -1
    for row in csv_reader:
        curr = row["video"]
        start_time = int(row["timestamp start"])
        end_time = int(row["timestamp end"]) + 1

        if prev == -1:
            initialiseVar(curr)
        elif prev != curr:
            initialiseVar(curr)
        prev = curr

        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))
        end_frame = int(end_time * cap.get(cv2.CAP_PROP_FPS))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = OUTPUT_VIDEO_NAME + str(FRAME_NUM) + ".mp4"

        print(VIDEO_FOLDER, VIDEO_FILE_PATH, output)

        out = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            success, frame = cap.read()
            if success:
                out.write(frame)
                # print("success")
            else:
                break

        out.release()
        cap.release()
        shutil.move(output, VIDEO_FOLDER + "/")
        FRAME_NUM += 1

for folder in FOLDERS:
    # change the to file_path based on the date make the dir first before running this script
    shutil.move(folder, "misdetection_output_210324/")
