from pathlib import Path

from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
# model = YOLO("yolov8s.yaml")
model = YOLO(
    "weights/yolov8n_150624_imgsz_640_1.pt"
)  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load(
#     "models/YOLO/yolov8n.pt"
# )  # build from YAML and transfer weights

# Load hyperparameters
# hyperparameters_file_path = (
#     Path().home()
#     / "rm_cv"
#     / "hyperparameters/yolov8n_310524_1/best_hyperparameters_1.yaml"
# )

# Set starting tuning hyperparameters
hyperparameters = {
    "fliplr": 0.0,  # Disable flipping left-right as it does not make sense for numbers
    "hsv_s": 0.8,
    "shear": 10,
    "perspective": 0.0006,
}

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
    resume=False,
    # cfg=hyperparameters_file_path,
    **hyperparameters
)
