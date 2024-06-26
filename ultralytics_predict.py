import os
from pathlib import Path

from natsort import natsorted
from ultralytics import YOLO

version = "yolov8n_120624_imgsz_640_1"
model = YOLO(f"weights/{version}.pt")

#####################################
#   Define folders to predict for   #
#####################################

# Predict for a list of directories
folderNames = [
    # "car_detection_images",
    # "RMUL2023 Sentry",
]

# Predict for subdirectories at a certain depth relative to `root_dir`
dir_name = "RMUL2023 Sentry/d1_aruw"
root_dir = Path(f"datasets/raw") / dir_name
depth = 1

for root, dirs, files in os.walk(root_dir):
    if len(Path(root).parents) - len(root_dir.parents) == depth:
        folderNames.append(str(Path(root).relative_to("datasets/raw")))

#######################
#   Prediction code   #
#######################

for folder_name in natsorted(folderNames):
    source = Path(f"datasets/raw/{folder_name}")

    # folderName = "input"
    # source = Path(f"datasets/raw/{folderName}/")

    project = f"validation/{folder_name}"
    vidName = f"{version}"

    # results would be a generator which is more friendly to memory by setting stream=True
    try:
        results = model.predict(
            source=source,
            show=False,
            save_txt=True,
            project=project,
            name=f"{vidName}",
            save=True,
            exist_ok=True,
            stream=True,
            show_labels=True,
        )
    except:
        pass

    for result in results:
        # print(result.boxes.xywhn)
        # print(result.boxes.cls.tolist()[0] == 0)
        continue
