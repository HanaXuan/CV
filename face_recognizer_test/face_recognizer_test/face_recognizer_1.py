import cv2
import numpy as np
import os
import json

recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 1,
                                                neighbors = 8,
                                                grid_x = 8,
                                                grid_y = 8,)
recognizer.read('trainer.yml')
with open('labels.json', 'r') as fp:
    labels = json.load(fp)

face_cascade_Path = "cascade.xml"
font = cv2.FONT_HERSHEY_SIMPLEX

faceCascade = cv2.CascadeClassifier(face_cascade_Path)

id = 0
#Video Capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
# Min Height and Width for the  window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print("id",id)
        print("confidence",confidence)
        if (confidence < 100):
            print ("id",id)
            id = labels[str(id)]
            confidence = "  {0}%".format(round(100-confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        else:
            # Unknown Face
            id = "Who are you ?"
            confidence = "  {0}%".format(round(100-confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        
        

    cv2.imshow('camera', img)
    # Escape to exit the webcam / program
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
