# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:59:00 2022

@author: Cristian
"""
import numpy as np
from sklearn import svm
import joblib
import sounddevice as sd 
from scipy.io.wavfile import write 
import wavio as wv 

############################ Entrenamiento del modelo #########################

patrones = list(np.loadtxt('Matriz_Patrones2.txt'))

l = list(np.transpose(np.concatenate((np.full((1,50),1),np.full((1,50),2),np.full((1,50),3),
                      np.full((1,50),4), np.full((1,50),5)),axis=1)))

y = []
for i in l:
    y.append(int(i))

clf = svm.SVC()
clf.fit(patrones,y)


#joblib.dump(clf, 'Modelo2.joblib') 

######################################### GRABAR AUDIO ########################

# f = 44400
# d = 0.8  
# recording = sd.rec(int(d * f), samplerate = f, channels = 2) 
  
# sd.wait() 
  
# write("Letra_N.wav", f, recording) 
# wv.write("Letra_N.wav", recording, f, sampwidth=2)
