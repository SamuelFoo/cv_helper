import os
from pathlib import Path

from ultralytics import YOLO

version = "front_yolov8n_210324_1"
model = YOLO(f"weights/{version}.pt")

# Test Data
# folderName = "old-bags"
# source = Path(f"datasets/processed/{folderName}/images/test")

folderNames = [
    # "pooltests/271223/main_gate_detection",
    # "pooltests/040124/qual_3.bag/right.raw",
    # "pooltests/300124/6.bag",
    # "pooltests/270224/2.bag",
    # "pooltests/270224/4.bag",
    # "pooltests/120324/21.bag",
    # "sauvc_v2_bboxes_yolo_front/images",
]

root_dir = "pooltests/210324/"
for sub_dir in list(os.walk(f"datasets/raw/{root_dir}"))[0][1]:
    folderNames.append(root_dir + sub_dir)

for folder_name in folderNames:
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
    )

    for result in results:
        # print(result.boxes.xywhn)
        # print(result.boxes.cls.tolist()[0] == 0)
        continue
