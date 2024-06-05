from pathlib import Path

from ultralytics import YOLO

# Initialize the YOLO model
# model = YOLO("weights/yolov8n_310524_1.pt")
model = YOLO("yolov8n.yaml").load(
    "models/YOLO/yolov8n.pt"
)  # build from YAML and transfer weights

# Set starting tuning hyperparameters
hyperparameters = {
    "fliplr": 0.0,  # Disable flipping left-right as it does not make sense for numbers
    "shear": 10,
    "perspective": 0.0005,
}

# Load hyperparameters
hyperparameters_file_path = (
    Path().home()
    / "rm_cv"
    / "hyperparameters/yolov8n_310524_1/best_hyperparameters_1.yaml"
)

# Tune hyperparameters
model.tune(
    data="front_cam.yaml",
    epochs=50,
    iterations=300,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
    batch=-1,
    **hyperparameters
    # cfg=hyperparameters_file_path,
)
