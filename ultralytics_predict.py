import os
from pathlib import Path

from ultralytics import YOLO

version = "yolov8n_250524_1"
model = YOLO(f"weights/{version}.pt")

# folderNames = ["dungeon/051624"]
folderNames = ["RMUL_2023_NA"]

# Predict for subdirectories.
# root_dir = "pooltests/210324/"
# for sub_dir in list(os.walk(f"datasets/raw/{root_dir}"))[0][1]:
#     folderNames.append(root_dir + sub_dir)

for folder_name in folderNames:
    source = Path(f"datasets/raw/{folder_name}")

    # folderName = "input"
    # source = Path(f"datasets/raw/{folderName}/")

    project = f"validation/{folder_name}"
    vidName = f"{version}"

    # results would be a generator which is more friendly to memory by setting stream=True
    results = model.predict(
        source=source,
        show=True,
        save_txt=True,
        project=project,
        name=f"{vidName}",
        save=True,
        exist_ok=True,
        stream=True,
    )

    for result in results:
        # print(result.boxes.xywhn)
        # print(result.boxes.cls.tolist()[0] == 0)
        continue
