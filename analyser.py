import shutil
import zipfile
from pathlib import Path
from typing import Callable, Generator, List

import cv2
import imutils
import pandas as pd

from cv_helper.helper import YOLOToCOCOBox, truncate_video

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


def unzip_yolo_files(folder: Path):
    for zip_file_path in folder.glob("*yolo.zip"):

        labels_path = Path(folder / "labels")
        labels_path.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(labels_path)


def exclusive_glob(
    root_dir: Path, pattern: str, exclude_patterns: List[str] = []
) -> Generator[Path, None, None]:
    """Glob that excludes patterns.

    Args:
        root_dir (Path): Root directory.
        pattern (str): Glob pattern.
        exclude_patterns (List[str], optional): List of glob patterns to exclude. Defaults to [].

    Returns:
        Generator[Path, None, None]: Generator glob object.
    """
    exclude_paths = set().union(
        *[set(root_dir.glob(exclude_pattern)) for exclude_pattern in exclude_patterns]
    )
    paths: List[Path] = set(root_dir.glob(pattern)) - exclude_paths
    return paths


def group_files_into_folder(
    root_dir: Path,
    file_paths: List[Path],
    get_folder_name_fn: Callable[[str], str] = lambda x: x.split("_")[0],
) -> None:
    """Groups files with similar names into the same folder.
    Files with the same output when passed into `folder_name_fn`
    will be grouped into the same folder.

    Args:
        root_dir (Path): Root directory.
        file_paths (List[Path], optional): List of file paths.
        get_folder_name_fn (Callable[[str], str], optional):
            Function that gets the folder name from the file name.
            Defaults to a function that splits by `_` and returns the first part.
    """

    for path in file_paths:
        file_name = path.name

        if path.is_file():
            folder_name = get_folder_name_fn(file_name)
            Path(root_dir / folder_name).mkdir(exist_ok=True)
            shutil.move(root_dir / file_name, root_dir / folder_name / file_name)


def group_files_and_unzip_yolo(
    root_dir: Path,
    file_paths: List[Path],
    get_folder_name_fn: Callable[[str], str] = lambda x: x.split("_")[0],
) -> None:
    group_files_into_folder(root_dir, file_paths, get_folder_name_fn)

    # Unzip yolo zip files
    for folder in root_dir.glob("*"):
        if folder.is_dir():
            unzip_yolo_files(folder)


def copyFiles(currPaths, datasetType):
    if len(currPaths) == 0:
        return

    saveDir = Path(currPaths[0].replace("root", datasetType)).parent
    saveDir.mkdir(parents=True, exist_ok=True)

    for src in currPaths:
        dst = src.replace("root", datasetType)
        shutil.copy2(src, dst)


def replace_path_string(path: Path, old_str: str, new_str: str) -> Path:
    """Replaces substring in Path object.

    Args:
        path (Path): input path
        old_str (str): old substring to be replaced
        new_str (str): new substring to replace with

    Returns:
        Path: modified path
    """
    return Path(str(path).replace(old_str, new_str))


def split_video_by_csv(dir_path: Path, video_path: Path, csv_path: Path):
    df = pd.read_csv(csv_path, index_col=None)
    start_times = df["Start Time"]
    end_times = df["End Time"]

    for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        output_video_path = dir_path / f"output_{i}.mp4"
        truncate_video(video_path, output_video_path, start_time, end_time)


#########################
#    Video to Images    #
#########################

# colors_hex = [
#     "#fa3253",
#     "#ff6037",
#     "#2a7dd1",
#     "#ff007c",
#     "#fafa37",
#     "#3d3df5",
#     "#cc3366",
# ]
colors_hex = [
    "#8c78f0",
    "#33ddff",
    "#3d3df5",
    "#5986b3",
    "#2a7dd1",
    "#f078f0",
    "#fa3253",
    "#ff007c",
    "#ff6037",
    "#ff355e",
]
colors_rgb = list(map(hex_to_rgb, colors_hex))
colors_bgr = list(map(lambda t: t[::-1], colors_rgb))


def generate_save_dir(dir_path: Path) -> tuple[Path, Path]:
    """Creates and return corresponding `save_dir` in `datasets/processed`.

    Args:
        dir_path (Path): `dir_path` in `datasets/raw`

    Returns:
        tuple[Path, Path]: `imgRootDir` and `labelRootDir`
    """
    save_dir_string = str(dir_path).replace("datasets/raw", "datasets/processed")
    saveDir = Path(save_dir_string)
    imgRootDir: Path = saveDir / "images" / "root"
    labelRootDir: Path = saveDir / "labels" / "root"
    imgRootDir.mkdir(parents=True, exist_ok=True)
    labelRootDir.mkdir(parents=True, exist_ok=True)
    return imgRootDir, labelRootDir


def video_to_images(
    dirPath: Path,
    videoName: str,
    subsample: int,
    frame_bounds: tuple[int, int],
    display: bool = True,
    display_interval: int = 5,
):
    """Export video as a YOLO dataset given YOLO labels.

    Args:
        dirPath (Path): _description_
        videoName (str): _description_
        subsample (int): _description_
        frame_bounds (tuple[int, int]): _description_
        display (bool, optional): _description_. Defaults to True.
        display_interval (int, optional): _description_. Defaults to 5.
    """
    vidPath = dirPath / videoName
    labelPaths = list(dirPath.glob("labels/obj_train_data/*.txt"))
    framesWithLabels = [int(path.stem.strip("frame_")) for path in labelPaths]

    imgRootDir, labelRootDir = generate_save_dir(dirPath)

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
