from ultralytics import YOLO

#load model
modelo_autos = YOLO("yolov8n.pt")
modelo_patentes = YOLO('license_plate_detector.pt')

#load video

