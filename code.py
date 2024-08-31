import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imageFile'
images = []
classNames = []

myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# image encoding 
def findEncodings(images):
    encodeList = []
    # print(images)
    for img in images:
        # print("+++++",img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding completed')

cap = cv2.VideoCapture(0)

while(True):
    rect,frame = cap.read()
    imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame =face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace , faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print("++++",matches)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print("----",faceDis)

        matchIndex = np.argmin(faceDis)
        print(matchIndex)
        # print(faceDis)
        # print("------",matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4 ,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            print(name)
            print("ACCESS GRANTED ")

    cv2.imshow('webcam results',frame)
    k = cv2.waitKey(2)
   # if k == 27:  # close on ESC key
       # cv2.destroyAllWindows()
    