from pathlib import Path

import supervision as sv
from cv_helper.analyser import get_bad_label_paths


def load_yolo_dataset(
    dataset_name: str, dataset_type: str, check_bad_labels: bool = False
):
    """
    Load a YOLO dataset from the specified path.
    """
    base_path = f"datasets/raw/{dataset_type}/{dataset_name}"

    if check_bad_labels:
        image_dir = Path(base_path, "images")
        label_dir = Path(base_path, "labels")
        get_bad_label_paths(image_dir=image_dir, label_dir=label_dir)

    ds = sv.DetectionDataset.from_yolo(
        data_yaml_path=f"datasets/raw/{dataset_type}/{dataset_name}/data.yaml",
        images_directory_path=f"datasets/raw/{dataset_type}/{dataset_name}/images",
        annotations_directory_path=f"datasets/raw/{dataset_type}/{dataset_name}/labels",
        is_obb=False,
    )

    return ds


def merge_datasets_by_name(
    dataset_names: list[str], dataset_type: str
) -> sv.DetectionDataset:
    """
    Merge datasets by name.
    """
    datasets = []

    for dataset_name in dataset_names:
        ds = load_yolo_dataset(dataset_name=dataset_name, dataset_type=dataset_type)
        datasets.append(ds)

    return sv.DetectionDataset.merge(datasets)


def get_identity_dict(keys: list[str]) -> dict[str, str]:
    """
    Create an identity mapping dictionary for the given keys.
    """
    return {key: key for key in keys}
