import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read() # frame olarak goruntuyu aldık
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,0,0])
    upper_yellow = np.array([40,255,255])
    mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    mask2= cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    img = cv2.medianBlur(res,5)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    imga = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(imga, cv2.HOUGH_GRADIENT, 1, 20,                  
                     param1=50, param2=30, minRadius=20, maxRadius=30)
    if circles is not None:
     circles = np.uint16(np.around(circles))
     for i in circles[0, :]:
         #cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
         #cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
           cv2.circle(img, (i[0], i[1]), i[2], (90, 255, 0), 2)
           cv2.circle(img, (i[0], i[1]), 2, (100, 0, 255), 3)
    #im=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)       
    cv2.imshow('res',img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
