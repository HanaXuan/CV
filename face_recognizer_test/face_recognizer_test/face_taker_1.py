import numpy as np
import cv2
import os

# Check if folder exists
face_id = input('name: ')
num = int(input('number of frame to record:'))
path = os.path.join('images' , face_id)
if not os.path.exists(path):
    os.makedirs(path)
    count = 0
    max = num
else:
    count =len(os.listdir(path))
    max = count + num

faceCascade = cv2.CascadeClassifier('cascade.xml')
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier('cascade.xml')
# For each person, enter one unique numeric face id
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

while(True):
    print(count, max)
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the images directory
        cv2.imwrite(os.path.join(path, str(count) + ".jpg"), gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    # Press Escape to end the program.
    k = cv2.waitKey(100) & 0xff
    if k < 30:
        break
    # Take 30 face samples and stop video. You may increase or decrease the number of
    # images. The more the better while training the model.
    elif count==max:
         break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()
