import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier(r'C:\Users\hp\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


eye_cascade = cv2.CascadeClassifier(r'C:\Users\hp\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_eye.xml')


smile_cascade = cv2.CascadeClassifier(r'C:\Users\hp\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        for (sx, sy, sw, sh) in smile: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 






        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
