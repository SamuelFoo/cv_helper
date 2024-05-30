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

# Train the model
model.train(data="front_cam.yaml", epochs=200, imgsz=640, patience=0, batch=-1)
