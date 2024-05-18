from ultralytics import YOLO

weights_file_name = "front_yolov8n_070424_1.pt"
model = YOLO(f"weights/{weights_file_name}")

model.export(format="onnx", imgsz=(480, 640))
