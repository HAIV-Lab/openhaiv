from ultralytics import YOLO
def main():
    model = YOLO('yolov8s-obb.yaml').load('yolov8s-obb.pt')  # build from YAML and transfer weights
    model.train(data='att.yaml', epochs=100, imgsz=672, batch=4, workers=4)
if __name__ == '__main__':
    main()