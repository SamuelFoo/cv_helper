from ultralytics import YOLO

weights_file_name = "yolov8n_250524_4.pt"
model = YOLO(f"weights/{weights_file_name}")

model.export(format="onnx", imgsz=(480, 640))
