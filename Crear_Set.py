# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:27:01 2022

@author: Cristian
"""
import cv2
import os
import time

nombre = 'Dataset'
direccion = #'Dirección de ubicación'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print('Carpeta Creada: ', carpeta)
    os.makedirs(carpeta)
cap = cv2.VideoCapture(1)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Imagen", frame)
    time.sleep(2)
    cv2.imwrite(carpeta + "/F{}.jpg".format(count), frame)
    count = count + 1
    
    t = cv2.waitKey(1)
    if t == 27 or count >= 50:
        break
cap.release()
cv2.destroyAllWindows()