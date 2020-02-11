import cv2
cam = cv2.VideoCapture(1)
while True:
    ret, im =cam.read()
    cv2.imshow('im',im) 
     
    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

    
