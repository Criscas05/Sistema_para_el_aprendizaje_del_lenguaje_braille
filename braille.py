# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:47:24 2022

@author: Cristian
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

patrones = np.empty((250,525))

for o in range(1,251):

    img = cv2.imread('Dataset/F{}.jpg'.format(o))
    #img = cv2.imread('Dataset/F166.jpg')
    res = cv2.resize(img, dsize=(300, 225), interpolation=cv2.INTER_AREA)
    resG = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    
    canny = cv2.Canny(res,180,250)
    kernel = np.ones((2,2),np.uint8)
    canny = cv2.dilate(canny,kernel) 
    
    output = cv2.connectedComponentsWithStats(canny,4,cv2.CV_32S)
    numObj = output[0]
    et = output[1] #Etiquetas
    est = output[2] #Datos de cada etiqueta
    
    seg_mayor = max(est[1:,4])
    ind = np.where(est[1:,4]==seg_mayor)
    ind2 = int(ind[0])+1
    
    x = 0;
    for j in range(len(est[:,4])):
          if x != ind2 and x != 0:
            for i in range(et.shape[0]):
                for k in range(et.shape[1]):
                    if et[i,k] == x:
                        et[i,k] = 0               
          x = x + 1
    
    contours,_ = cv2.findContours(np.uint8(et), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    #drawing = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)    # VISUALIZAR
    #coun = cv2.drawContours(drawing,contours,-1, (255,255,0),1)           # VISUALIZAR
    
    hull = cv2.convexHull(cnt)
    puntosConvex = hull[:,0,:]
    m,n = et.shape
    ar = np.zeros((m,n))
    mascaraConvex = np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1))
    
    contours2,_=cv2.findContours(mascaraConvex,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt2 = contours2[0]
    #drawing1 = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)                  #VISUALIZAR
    #coun1 = cv2.drawContours(drawing1,contours2,-1, (255,255,0),1)                        #VISUALIZAR
    
    rect = cv2.minAreaRect(cnt2)
    box = np.int0(cv2.boxPoints(rect))
    ar = np.zeros((m,n))
    mascaraRect = np.uint8(cv2.fillConvexPoly(ar,box,1))
    
    # ############################################ HOMOGRAFIA ############################
    
    vertices = cv2.goodFeaturesToTrack(mascaraRect,4,0.01,20)
    
    vertices = vertices[:,0,:]
    
    x = vertices[:,0]
    y = vertices[:,1]
    
    xo = np.sort(x)
    yo = np.sort(y)
    
    xn = np.zeros((1,4))
    yn = np.zeros((1,4))
    
    xn = (x==xo[2])*n+(x==xo[3])*n
    yn = (y==yo[2])*m+(y==yo[3])*m
    
    verticesN = np.zeros((4,2))
    verticesN[:,0] = xn
    verticesN[:,1] = yn
    
    vertices = np.int64(vertices)
    verticesN = np.int64(verticesN)
    
    for ver in verticesN:
        if ver[0] != 0 and ver[0] != n:
            ver[0] = ver[0]/2
        if ver[1] != 0 and ver[1] != m:
            ver[1] = ver[1]/2
    
    h,_ = cv2.findHomography(vertices,verticesN)
    
    im2 = cv2.warpPerspective(res,h,(n,m))
    
    ############################################################################
    ycbcr = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb) 
    
    v = ycbcr[:,:,::-1]
    
    for i in range(len(res[:,1])):
        for k in range(len(res[1,:])):
            cr = v[i,k,1]
            cb = v[i,k,2]
            if ~(cr < 110 and cb > 50):
                v[i,k,0] = 255
                v[i,k,1] = 120
                v[i,k,2] = 120
    
    dev = cv2.cvtColor(v, cv2.COLOR_YCrCb2BGR)
    gray = cv2.cvtColor(dev, cv2.COLOR_BGR2GRAY)
    img_fil = cv2.GaussianBlur(gray,(11,11),0)
    b1,b2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bw = np.uint8(~b2/255)
    
    #cleaned = morphology.remove_small_objects(np.array(bw, dtype=bool),min_size=64, connectivity=2)
    #output = cv2.connectedComponentsWithStats(np.uint8(cleaned),4,cv2.CV_32S)
    
    output = cv2.connectedComponentsWithStats(bw,4,cv2.CV_32S)
    et = output[1]
    est = output[2]
    
    x = 0;
    for j in est[:,4]:
        if j < 500:
            for i in range(et.shape[0]):
                for k in range(et.shape[1]):
                    if et[i,k] == x:
                        et[i,k] = 0
        x = x + 1
    
    relleno = ndimage.binary_fill_holes(et).astype(int)
    
    
    contours3,_=cv2.findContours(np.uint8(relleno),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    res_final = np.zeros((m,n))
    for i in range(len(contours3)):
        
        co = contours3[i]
        hull2 = cv2.convexHull(co)
        puntosConvex2 = hull2[:,0,:]
        ar = np.zeros((m,n))
        final = np.uint8(cv2.fillConvexPoly(ar,puntosConvex2,1))
        
        res_final = final + res_final
    
    ################# Extracción de patrones(Integral Proyectiva)##############
    g = 0
    g2 = 0;
    v = []
    v2 = [];
    
    ############ POR FILAS ##############
    for i in range(relleno.shape[0]):
        for k in range(relleno.shape[1]):
            g = g + res_final[i,k]
        v.append(g/relleno.shape[0])
        g = 0
    ############ POR COLUMNAS ###########
    for i2 in range(relleno.shape[1]):
        for k2 in range(relleno.shape[0]):
            g2 = g2 + res_final[k2,i2]
        v2.append(g2/relleno.shape[1])
        g2 = 0;
    
    zona = v+v2
        
    patrones[o-1,:] = zona
    
# np.savetxt('Matriz_Patrones.txt',patrones)
