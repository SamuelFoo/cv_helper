import shutil
import zipfile
from pathlib import Path
from typing import Callable, Generator, List

import cv2
import imutils
import numpy as np
import pandas as pd
from natsort import natsorted

from cv_helper.helper import YOLOToCOCOBox, getLabelPaths, truncate_video
from supervision.dataset.formats.yolo import yolo_annotations_to_detections, _with_mask
import supervision as sv

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


def copyFiles(curr_paths: List[Path], dataset_type: str):
    if len(curr_paths) == 0:
        return

    save_dir = Path(str(curr_paths[0]).replace("root", dataset_type)).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    for src in curr_paths:
        dst = str(src).replace("root", dataset_type)
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


#############################
#    Process Raw Dataset    #
#############################

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
    "#aaf0d1",
    "#5986b3",
    "#34d1b7",
    "#733380",
    "#fa7dbb",
    "#ff007c",
    "#ff6037",
    "#ff355e",
    "#b83df5",
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


def video_to_yolo_dataset(
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

                winname = str(dirPath / videoName)
                cv2.namedWindow(winname)  # Create a named window
                cv2.moveWindow(winname, 0, 200)
                cv2.imshow(winname, frame)

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


def images_to_yolo_dataset(
    dir_path: Path, remap_fn: Callable[[str], str] = lambda x: x
):
    # Make save directories
    img_root_dir, label_root_dir = generate_save_dir(dir_path)

    img_paths = natsorted(list(dir_path.rglob("*.[jp][pn]g")))
    label_paths = list(dir_path.glob("labels/obj_train_data/*.txt"))
    stems_with_labels = [path.stem for path in label_paths]

    for img_path, label_path in zip(img_paths, label_paths):
        # Skip if label does not exist
        if not img_path.stem in stems_with_labels:
            continue

        # Copy image only if there is a label
        shutil.copy2(img_path, img_root_dir / img_path.name)

        # Copy label and remap classes
        remap_class_labels(
            label_path,
            (label_root_dir / Path(label_path).stem).as_posix()
            + ".txt",  # with_suffix does not handle paths with "." well
            remap_fn=remap_fn,
        )


def remap_class_labels(
    src_file_path: Path, dst_file_path: Path, remap_fn: Callable[[str], str]
):
    # Get old detections
    with open(src_file_path, "r") as f:
        file_str = f.read()

    detections = [detection.split(" ") for detection in file_str.strip().split("\n")]
    new_detections = []

    for detection in detections:
        # Remap class
        cls = detection[0]
        new_detection = detection.copy()
        new_detection[0] = remap_fn(cls)
        new_detections.append(new_detection)

    # Convert to new detection
    new_detections = [" ".join(d) + "\n" for d in new_detections]

    # Write to file
    with open(dst_file_path, "w") as f:
        f.writelines(new_detections)


def remap_detections_class_labels(
    dataset: sv.DetectionDataset,
    detections: sv.Detections,
    remap_dict: dict[str, str],
    remapped_classes: list,
) -> sv.Detections:
    # Source: https://github.com/roboflow/supervision/issues/1778#issue-2799947071
    # Remove predicted classes not in keys to remap
    detections_class_names = [
        dataset.classes[class_id] for class_id in detections.class_id
    ]
    # Conversion to list is needed for np.isin to work
    detections = detections[np.isin(detections_class_names, list(remap_dict.keys()))]

    # Remap class names
    remapped_detections_class_names = [
        dataset.classes[class_id] for class_id in detections.class_id
    ]

    # Remap Class IDs based on Class names
    detections.class_id = np.array(
        [remapped_classes.index(name) for name in remapped_detections_class_names]
    )

    return detections


def remap_dataset_class_labels(
    dataset: sv.DetectionDataset, remap_dict: dict[str, str]
):
    # Check if all mapped values are within the dataset classes
    if not all([value in dataset.classes for value in remap_dict.values()]):
        raise ValueError("All mapped values must be in dataset classes")

    remapped_annotations = []
    remapped_classes = list(dict.fromkeys(remap_dict.values()))
    for path, detections in dataset.annotations.items():
        detections = remap_detections_class_labels(
            dataset, detections, remap_dict, remapped_classes
        )
        remapped_annotations.append((path, detections))

    remapped_annotations = dict(remapped_annotations)
    remapped_dataset = sv.DetectionDataset(
        classes=remapped_classes,
        images=dataset.annotations.keys(),
        annotations=remapped_annotations,
    )

    return remapped_dataset


def split_train_valid_test(base_dir: Path, proportions_dict: dict):
    """Create train, valid, and test directories from the root directory.

    Resultant base_dir will have the following structure:
    ```
    base_dir
    ├──images
    │  ├──root
    │  ├──train
    │  ├──valid
    │  └──test
    └──labels
    │  ├──root
    │  ├──train
    │  ├──valid
    │  └──test
    ```

    Args:
        base_dir (Path): Base directory containing the images and labels.
        base_dir should have the following structure:
        ```
        base_dir
        ├──images
        │  ├──root
        │  ├──(train)
        │  ├──(valid)
        │  └──(test)
        └──labels
        │  ├──root
        │  ├──(train)
        │  ├──(valid)
        │  └──(test)
        ```
        with optional train, valid, and test directories.
        Existing train, valid, and test directories will be replaced.
        proportions_dict (dict): Proportions of dataset in train, valid and test directories.
    """

    def split_integer(total: int, proportions: list) -> list:
        """
        Split an integer as close as possible to given proportions.

        Args:
            total (int): The integer to split.
            proportions (dict): Proportions as floats. Must sum to 1.

        Returns:
            list: List of split integers.
        """
        assert sum(proportions) == 1, "Proportions must sum to 1"
        assert all(
            proportion >= 0 for proportion in proportions
        ), "Proportions must be non-negative"

        # Calculate the initial split
        split_values = [int(total * prop) for prop in proportions]

        # Adjust the split to ensure the sum is equal to the total
        remainder = total - sum(split_values)
        for i in range(remainder):
            split_values[i % len(proportions)] += 1

        return split_values

    def replace_dir(dst_dir: Path, file_paths: List[Path], dataset_type: str):
        shutil.rmtree(dst_dir, ignore_errors=True)
        copyFiles(file_paths, dataset_type)

    # Reset each time so that number of processed dirs before does not affect choice.
    np.random.seed(314159)

    img_paths = np.array(natsorted((base_dir / "images" / "root").glob("*")))

    dataset_types = list(proportions_dict.keys())
    dataset_props = list(proportions_dict.values())
    dataset_counts = split_integer(len(img_paths), dataset_props)

    for dataset_type, dataset_count in zip(dataset_types, dataset_counts):
        if dataset_count == 0:
            continue

        # Replace images
        dataset_img_paths = np.random.choice(img_paths, dataset_count, replace=False)
        dataset_img_dir = base_dir / "images" / dataset_type
        replace_dir(dataset_img_dir, dataset_img_paths, dataset_type)

        # Replace labels
        dataset_label_paths = getLabelPaths(dataset_img_paths)
        dataset_label_dir = base_dir / "labels" / dataset_type
        replace_dir(dataset_label_dir, dataset_label_paths, dataset_type)

        # Remove paths from pool
        img_paths = np.setdiff1d(img_paths, dataset_img_paths)


def get_bad_label_paths(image_dir: Path, label_dir: Path) -> None:
    image_files = list(Path(image_dir).glob("*.jpg")) + list(
        Path(image_dir).glob("*.png")
    )

    for image_path in image_files:
        label_path = label_dir / image_path.with_suffix(".txt").name

        if not Path(label_path).exists():
            continue

        try:
            image = cv2.imread(image_path)
            resolution_wh = (image.shape[1], image.shape[0])  # width, height

            with open(label_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            with_masks = _with_mask(lines=lines)
            _ = yolo_annotations_to_detections(
                lines=lines,
                resolution_wh=resolution_wh,
                with_masks=with_masks,
                is_obb=False,
            )

        except Exception as e:
            print(f"\n❌ Error in file: {label_path}")
            print(f"   Reason: {e}")


#####################################
#    Convert Ultralytics Formats    #
#####################################


def convert_predicted_labels_to_yolo_dataset(
    root_dir: Path, new_label_name_generator: Callable[[str], str], image_ext: str
) -> None:
    """Given labels generated by `ultralytics_predict.py`, convert them into a
    yolo dataset that can be read by CVAT.

    Args:
        root_dir (Path): Directory containing the input images / video and labels.
        This will be the same directory where the dataset is created.
        new_label_name_generator (Callable[[str], str]): Function that names the label
        files by a specific rule.
        image_ext (str): File extension of the images
    """

    labels_dir = root_dir / "predicted_labels"
    dataset_dir = root_dir / "predicted_labels_for_cvat_import"
    new_labels_dir = dataset_dir / "obj_train_data"
    new_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_dir / "train.txt", "w") as train_txt_file:
        for label_path in natsorted(labels_dir.glob("*.txt")):
            new_label_name: str = new_label_name_generator(label_name=label_path.name)
            new_label_path = new_labels_dir / new_label_name
            shutil.copy2(label_path, new_label_path)
            train_txt_file.write(
                f"data/obj_train_data/{new_label_path.stem}{image_ext}\n"
            )

    shutil.copy2("cv_helper/cvat_yolo_metadata/obj.data", dataset_dir)
    shutil.copy2("cv_helper/cvat_yolo_metadata/obj.names", dataset_dir)
    shutil.make_archive(root_dir / dataset_dir.stem, "zip", dataset_dir)


#######################
#    Sanity Checks    #
#######################


def check_train_valid_split(root_dir: Path) -> None:
    """Check if the train and valid directories have the same number of files as the root directory.

    Args:
        root_dir (Path): Root directory.
    """

    def get_subdirs(dir_path: Path) -> List[Path]:
        return [sub for sub in dir_path.iterdir() if sub.is_dir()]

    # Get list of all directories
    all_dirs = [d for d in root_dir.glob("**/") if d.is_dir()]

    # Filter directories that have subdirectories but no further subdirectories
    # These are the "images" and "labels" directories
    result_dirs: List[Path] = []
    for d in all_dirs:
        subdirs = get_subdirs(d)
        if subdirs and all(not list(get_subdirs(sub)) for sub in subdirs):
            result_dirs.append(d)

    for result_dir in result_dirs:
        root_dir = result_dir / "root"
        train_dir = result_dir / "train"
        valid_dir = result_dir / "valid"
        assert len(list(root_dir.glob("*"))) > 0
        assert len(list(root_dir.glob("*"))) == len(list(train_dir.glob("*"))) + len(
            list(valid_dir.glob("*"))
        ), result_dir
