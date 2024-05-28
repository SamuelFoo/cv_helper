import os
from pathlib import Path

from natsort import natsorted
from ultralytics import YOLO

version = "yolov8n_250524_5"
model = YOLO(f"weights/{version}.pt")

#####################################
#   Define folders to predict for   #
#####################################

# folderNames = ["dungeon/051624"]
# folderNames = ["RMUL_2023_NA"]
folderNames = []

# Predict for subdirectories at a certain depth relative to `root_dir`
dir_name = "RMUL_2023_NA/"
root_dir = Path(f"datasets/raw") / dir_name
depth = 2

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
    results = model.predict(
        source=source,
        show=False,
        save_txt=True,
        project=project,
        name=f"{vidName}",
        save=True,
        exist_ok=True,
        stream=True,
        show_labels=False,
    )

    for result in results:
        # print(result.boxes.xywhn)
        # print(result.boxes.cls.tolist()[0] == 0)
        continue
