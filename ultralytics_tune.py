from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("weights/yolov8n_310524_1.pt")

hyperparameters = {
    "fliplr": 0.0,  # Disable flipping left-right as it does not make sense for numbers
}

# Tune hyperparameters on for 100 epochs
model.tune(
    data="front_cam.yaml",
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
    batch=-1,
    **hyperparameters,
)
