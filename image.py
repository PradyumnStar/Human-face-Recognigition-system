import cv2 as cv 
import numpy as np
import os
cap = cv.VideoCapture(0)

result,frame=cap.read()
name=input('Enter File Name')
path=f'C:\PROJECT\imageFile'

if result:
    cv.imshow('Image',frame)
    
    cv.imwrite(os.path.join(path,name+'.jpeg'),frame)
    k = cv.waitKey(0) & 0xFF
    if k == 27:  # close on ESC key
        cv.destroyAllWindows()
else:
    print("No Image")