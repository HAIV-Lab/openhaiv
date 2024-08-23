from ultralytics import YOLO
model = YOLO("-") #训练好的.pt文件路径
results = model("-", save=True) #需要推理的图像路径