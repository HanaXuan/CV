import cv2
import numpy as np
from PIL import Image
import os
import json

#Directory path name where the face images are stored.
path = 'images'
recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 1,
                                                neighbors = 8,
                                                grid_x = 8,
                                                grid_y = 8,)

#Haar cascade file
detector = cv2.CascadeClassifier("cascade.xml");

def getImagesAndLabels(path):
    faces = []
    labels = {}
    ids = []
    num_name = 0
    for fold in os.listdir(path):
        print("fold",fold)
        labels[num_name] = fold.split("_")[0]
        for file in os.listdir(os.path.join(path, fold)):
            link = os.path.join(path, fold, file)
            img = cv2.imread(link, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(num_name)
            print("link",link)
        num_name +=1

    return ids, faces, labels

print ("\n[INFO] Training faces...")
ids,faces, labels = getImagesAndLabels(path)
ids = np.array(ids, dtype=np.int32)
#labels = cv2.UMat(labels)
recognizer.train(faces, ids)
# Save the model into the current directory.
recognizer.write('trainer.yml')
print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
print("ids",ids)
print("labels",labels)
with open('labels.json', 'w') as fp:
    json.dump(labels, fp)
def trained_names():
    global labels
    return labels