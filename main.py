from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import *

mot_tracker = Sort()

#cargar modelos
modelo_autos = YOLO("yolov8n.pt")
modelo_patentes = YOLO('license_plate_detector.pt')

#cargar video
video = cap = cv2.VideoCapture('./VideoAutos.MOV')

vehiculos = [2, 3, 5, 7]

#leer frames
frame_number = -1 
ret = True
while ret:
    frame_number += 1
    ret, frame = video.read()
    if ret and frame_number <10:
     
        #detect vehicles
        detection = modelo_autos(frame)[0]
        detecciones = []
        for detection in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehiculos: 
                detecciones.append([int(x1), int(y1), int(x2), int(y2), score])
                
        # traquear vehiculos
        track_ids = mot_tracker.update(np.array(detecciones))

        #detector de patentes
        detection_patentes = modelo_patentes(frame)[0]
        for detection_patentes in detection_patentes.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection_patentes
        
        #asignar patentes a vehiculos
                