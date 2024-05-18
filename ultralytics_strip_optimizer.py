from ultralytics.utils.torch_utils import strip_optimizer

name = "front_yolov8n_070424_1"
file_name = f"{name}_opt.pt"
strip_optimizer(f=f"weights/{file_name}", s=f"weights/{name}.pt")
