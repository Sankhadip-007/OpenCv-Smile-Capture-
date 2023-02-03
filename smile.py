import cv2
import numpy as np

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile1.xml')

def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray, 1.1, 8)
                                        # 1.1 : scaling factor
                                        # 8: number of nearest members
    for (x, y, w, h) in faces:
        roi_gray=gray[y:y+h,x:x+w] # extract the face from the gray image
        roi_frame=frame[y:y+h,x:x+w] # extract the face from the color image


        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (x, y, w, h) in smiles:
            cv2.rectangle(roi_frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
        if(len(smiles)>=1):
            cv2.putText(frame,"Sweet Smile",(20,40),cv2.FONT_HERSHEY_PLAIN,2,(0,255,250),2)
            cv2.imwrite('img1_saved.jpg',frame)
    return frame


while(True):
# or while(cap.isOpened()) :
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    smile=detect(gray,frame)

    frame=cv2.flip(frame,1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q') :
         # press 'q' to close
           break
cap.release()
cap.destroyAllWindows()