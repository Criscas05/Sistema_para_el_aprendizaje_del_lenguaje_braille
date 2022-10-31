# Sistema_para_el_aprendizaje_del_lenguaje_braille
Desarrollo de un sistema para el aprendizaje del lenguaje Braille mediante técnicas de tratamiento de imagen y visión artificial. 

El sistema pretende ser útil para el aprendizaje del lenguaje Braille, utilizarlo en colegios e incentivar el aprendizaje de lenguaje inclusivo  y esparcimiento de nuevo conocimiento.

La utilización está basada en cuatro scripts desarrollados en python

1) Sript 'Braille.py'
  Corresponde al desarrollo algoritmico necesario para el tratamiento de la imagen utilizado en la extracción de caracteristicas(patrones). Se tomaron 50 imagenes por clase,
  donde inicialmente se reconoce la letra F, G, H, I, J. Sin embargo, se pretende mejorar para el reconocimiento en palabras. Respecto a las técnicas implementadas están:
  la detección de bordes, mascara convexa, homografía, binarización, mejoramiento de bordes y finalmente la integral proyectiva como método de extraccion de patrones.

2) Script 'Crear_Set.py'
  Corresponde a la codificación necesaria para la generación del dataset, es libre, se pueden tomar la imágenes que se deseen. Sin embargo, se realiza el cargue de archivo
  Dataset.rar que corresponde al dataset utilizado.
  
3) Script 'Entrenamiento.py'
  Corresponde a la codificación realizada para el entrenamiento del modelo. Se carga el archivo .txt generado en el 'Braille.py' que corresponde a la matriz de patrones
  y se entrena el modelo.
  
4) Script 'Validación.py'
  Se codifica el algoritmo necesario para validar el comportamiento del modelo. Se implementa en tiempo real y se ejecutan pruebas de rendimiento.
