from ultralytics import YOLO

weights_file_name = "yolov8n_030624_imgsz_640_1.pt"
model = YOLO(f"weights/{weights_file_name}")

# imgsz = (384, 512)
imgsz = (480, 640)

model.export(format="onnx", imgsz=imgsz)
