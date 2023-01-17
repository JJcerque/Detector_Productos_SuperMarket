
#importar librerias

import torch
import cv2
import numpy  as np
import serial, time
from gtts import gTTS
from playsound import playsound
import webbrowser


#leer modelo


model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'Modelo\Super.pt')

#video capture
cap = cv2.VideoCapture(0)

while True:
    #lectura
    ret, frame = cap.read()

    #detecciones
    
    detect = model(frame)
    
    #FPS  
    Conteo = detect.pandas().xyxy[0].to_dict(orient='records')
    
    if len(Conteo) != 0:
        for result in Conteo:
            conf = result['confidence']
        if conf >= 0.80:
                cls = int(result['class'])
                xi = int(result['xmin'])
                yi = int(result['ymin'])
                xf = int(result['xmax'])
                yf = int(result['ymax'])
            
                if cls == 0:
                    Producto = gTTS("Spaguetti", lang = 'es')
                    Producto.save('sample.mp3')
                    webbrowser.open("sample.mp3")
                    time.sleep(3)

                if cls == 2:
                    Producto = gTTS("Aceite", lang = 'es')
                    Producto.save('sample.mp3')
                    webbrowser.open("sample.mp3")
                    time.sleep(3)
                    
                if cls == 3:
                    Producto = gTTS("Arroz", lang = 'es')
                    Producto.save('sample.mp3')
                    webbrowser.open("sample.mp3")
                    time.sleep(3)
                    
                if cls == 4:
                    Producto = gTTS("Fríjol", lang = 'es')
                    Producto.save('sample.mp3')
                    webbrowser.open("sample.mp3")
                    time.sleep(3)
                    
                if cls == 5:
                    Producto = gTTS("Atún", lang = 'es')
                    Producto.save('sample.mp3')
                    webbrowser.open("sample.mp3")
                    time.sleep(3)

    cv2.imshow('Detector', np.squeeze(detect.render()))


    t = cv2.waitKey(1) & 0xFF
    if t == ord("s"):
        break
    
cap.release()
cv2.destroyAllWindows
