from pathlib import Path

from ultralytics import YOLO


def ultralytics_predict(dir_path: Path, weights_path: Path):
    model = YOLO(weights_path)

    # results would be a generator which is more friendly to memory by setting stream=True
    results = model.predict(
        source=dir_path,
        show=True,
        save_txt=True,
        project=dir_path,
        save=True,
        exist_ok=True,
        stream=True,
        show_labels=True,
    )

    for result in results:
        continue
