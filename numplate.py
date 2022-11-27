# import the opencv library
from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract
import imutils



pytesseract.pytesseract.tesseract_cmd="C:/Program Files/Tesseract-OCR/tesseract.exe"

cascade= cv2.CascadeClassifier("C:/Users/Admin/Downloads/haarcascade_russian_plate_number.xml")


import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, img = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    #cv2.imshow("Test",edged)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            print(location)
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    #cv2.imshow("test",new_image)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    
    #nplate=cascade.detectMultiScale(cropped_image,5,7)

    cv2.imshow("test",cropped_image)
    #crop portion
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
