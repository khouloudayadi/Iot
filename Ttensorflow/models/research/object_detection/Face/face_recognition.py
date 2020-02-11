import cv2
import numpy as np
import time
import os 

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("trainer/")

cascadePath = "C:/Users/Hola/Desktop/Ttensorflow/models/research/object_detection/Face/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(1)
while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    recognizer.read('C:/Users/Hola/Desktop/Ttensorflow/models/research/object_detection/Face/trainer/trainer.yml')
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(confidence)
        if(confidence<= 81):
          if(Id == 1):
            Id = "badii {0:.2f}%".format(round(100 - confidence, 2))
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
            print("hello badii")
            os.system("python C:/Users/Hola/Desktop/VOICE.py")
            time.sleep(7)
            cam.release()
            cv2.destroyAllWindows()
        else:    
            Id = "inconnu{0:.2f}%".format(round(100 - confidence, 2))
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
            os.system("python C:/Users/Hola/Desktop/VOICE1.py")
            time.sleep(8)
    cv2.imshow('im',im) 
     
    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
