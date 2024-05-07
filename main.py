from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import *
from funciones import get_car, read_license_plate, write_csv

resultados = {}
mot_tracker = Sort()

#cargar modelos
modelo_autos = YOLO("yolov8n.pt")
modelo_patentes = YOLO('license_plate_detector.pt')

#cargar video
video = cv2.VideoCapture('VideoAutos.MOV')

vehiculos = [2, 3, 5, 7]

#leer frames
frame_number = -1 
ret = True
while ret:
    frame_number += 1
    ret, frame = video.read()
    if ret and frame_number <10:
        resultados[frame_number] = {} 
     
        #detectar vehiculos
        detection = modelo_autos(frame)[0]
        detecciones = []
        for detection in detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehiculos: 
                detecciones.append([x1, y1, x2, y2, score])


        # traquear vehiculos
        track_ids = mot_tracker.update(np.asarray(detecciones))

        #detector de patentes
        patente = modelo_patentes(frame)[0]
        for patente in patente.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = patente
        
        #asignar patentes a vehiculos
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(patente, track_ids)
        
        if car_id != -1:
            
            #obtener patente
            patente_cortada = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            #filtros (threshold)
            patente_cortada_gris = cv2.cvtColor(patente_cortada, cv2.COLOR_BGR2GRAY)
            _, licencia_cortada_treshold = cv2.threshold(patente_cortada_gris, 64, 255, cv2.THRESH_BINARY_INV)
            
            #leer la patente
            texto_patente, texto_patente_score = read_license_plate(licencia_cortada_treshold)
            
            if texto_patente is not None:
                resultados[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}, 
                                                    'license_plate': {'bbox': [x1, y1, x2, y2], 
                                                                    'text': texto_patente,
                                                                    'bbox_score': score,
                                                                    'text_score': texto_patente_score}}
            
# Resultados

write_csv(resultados, 'ResultadosLectorPatente.csv')