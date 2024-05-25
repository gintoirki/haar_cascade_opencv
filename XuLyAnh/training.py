import cv2 as cv
import os
import numpy as np

people = ['NgoTruongGiang','DoiXuanDat','DongPhucQuan','NguyenLeChinh']
DIR = r'D:\python_code\pythonProject\Coursera\XuLyAnh\img_data'

haar_cascades = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
features = []
labels = []
def feature_extraction():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascades.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

feature_extraction()
print('Training Done')

features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

