import shutil
import zipfile
from pathlib import Path

import cv2
import imutils

from cv_helper.helper import YOLOToCOCOBox

#################
#    General    #
#################


def check_frame_bound(frame_num, frame_bounds):
    for bound in frame_bounds:
        if bound[0] <= frame_num <= bound[1]:
            return True
    return False


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def process_misdetections(root_dir: Path):
    # Move output1_..._.mp4 and output1_..._.zip into output1/
    for path in root_dir.glob("*"):
        file_name = path.name

        if path.is_file():
            folder_name = path.stem.split("_")[0]
            Path(root_dir / folder_name).mkdir(exist_ok=True)
            shutil.move(root_dir / file_name, root_dir / folder_name / file_name)

    # Unzip yolo zip files
    for folder in root_dir.glob("*"):
        if folder.is_dir():
            for zip_file_path in folder.glob("*yolo.zip"):

                labels_path = Path(folder / "labels")
                labels_path.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(labels_path)


def copyFiles(currPaths, datasetType):
    if len(currPaths) == 0:
        return

    saveDir = Path(currPaths[0].replace("root", datasetType)).parent
    saveDir.mkdir(parents=True, exist_ok=True)

    for src in currPaths:
        dst = src.replace("root", datasetType)
        shutil.copy2(src, dst)


#########################
#    Video to Images    #
#########################

colors_hex = [
    "#fa3253",
    "#ff6037",
    "#2a7dd1",
    "#ff007c",
    "#fafa37",
    "#3d3df5",
    "#cc3366",
]
colors_rgb = list(map(hex_to_rgb, colors_hex))
colors_bgr = list(map(lambda t: t[::-1], colors_rgb))


def video_to_images(
    dirPath: Path, videoName, subsample, frame_bounds, display=True, display_interval=5
):
    vidPath = dirPath / videoName
    labelPaths = list(dirPath.glob("labels/obj_train_data/*.txt"))
    framesWithLabels = [int(path.stem.strip("frame_")) for path in labelPaths]

    save_dir_string = str(dirPath).replace("datasets/raw", "datasets/processed")
    saveDir = Path(save_dir_string)
    imgRootDir = saveDir / "images" / "root"
    labelRootDir = saveDir / "labels" / "root"
    imgRootDir.mkdir(parents=True, exist_ok=True)
    labelRootDir.mkdir(parents=True, exist_ok=True)

    frameCount = 0
    cap = cv2.VideoCapture(str(vidPath))
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        if (
            frameCount in framesWithLabels
            and frameCount % subsample == 0
            and check_frame_bound(frameCount, frame_bounds)
        ):
            idx = framesWithLabels.index(frameCount)
            labelPath = labelPaths[idx]
            shutil.copy2(labelPath, labelRootDir / labelPath.name)
            cv2.imwrite(str(imgRootDir / f"{labelPath.stem}.jpg"), frame)

            if display:
                f = open(labelPath, "r")
                boxes = f.read().strip().split("\n")

                classes = [int(box.strip().split()[0]) for box in boxes if box != ""]
                boxes = [
                    [float(num) for num in box.strip().split()[1:]]
                    for box in boxes
                    if box != ""
                ]

                for box, cls in zip(boxes, classes):
                    if box != []:
                        x, y, w, h = YOLOToCOCOBox(box, frame.shape[-2::-1])
                        cv2.rectangle(
                            frame,
                            (int(x), int(y)),
                            (int(x + w), int(y + h)),
                            colors_bgr[cls],
                            1,
                        )

                frame = imutils.resize(frame, height=500)
                cv2.imshow(str(dirPath / videoName), frame)
                key = cv2.waitKey(display_interval)
                if key == ord("q") or key == ord("Q"):
                    break

                if key == ord("p") or key == ord("P"):
                    while True:
                        key = cv2.waitKey(0)
                        if key == ord("p") or key == ord("P"):
                            break

        frameCount += 1

    cv2.destroyAllWindows()
