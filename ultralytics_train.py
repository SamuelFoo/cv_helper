from pathlib import Path

from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
# model = YOLO("yolov8s.yaml")
model = YOLO(
    "weights/yolov8n_290524_3.pt"
)  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load(
#     "models/YOLO/yolov8n.pt"
# )  # build from YAML and transfer weights

# Load hyperparameters
hyperparameters_file_path = (
    Path().home()
    / "rm_cv"
    / "hyperparameters/yolov8n_310524_1/best_hyperparameters_1.yaml"
)

# Train the model
model.train(
    data="front_cam.yaml",
    epochs=200,
    imgsz=640,
    patience=0,
    batch=-1,
    deterministic=False,
    profile=True,
    augment=True,
    cfg=hyperparameters_file_path,
)
