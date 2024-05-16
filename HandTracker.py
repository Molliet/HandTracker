import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import mediapipe as mp
import numpy as np

cx=0
cy=0
hcx=0
hcy=0


#Lining Landmarks
mp_drawing = mp.solutions.drawing_utils
#Style Line
mp_drawing_styles = mp.solutions.drawing_styles
#var for function
mphands = mp.solutions.hands

#Color Finder
myColorFinder = ColorFinder(False)

#Video Capture
cap=cv2.VideoCapture('bball.mp4')

hands = mphands.Hands()
hsvVals = {'hmin': 0, 'smin': 164, 'vmin': 120, 'hmax': 9, 'smax': 213, 'vmax': 155}




while True:
    data, image = cap.read()
    image_height, image_width, _ = image.shape
    imgClean = image


    #find and draw ball
    imgColor, Mask = myColorFinder.update(image,hsvVals)
    image , contours = cvzone.findContours(image, Mask, minArea=500)
    if contours:
        cx, cy = contours[0]['center']



    #find and draw hand
    #flip image
    image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
    #store results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #find landmarks in results
    if results.multi_hand_landmarks:
        #Line Landmarks as hand connections
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mphands.HAND_CONNECTIONS
            )
            for ids, landmrk in enumerate(hand_landmarks.landmark):
                if ids == 9:
                    hcx, hcy = landmrk.x * image_width, landmrk.y*image_height


    
    print(results.multi_handedness)
    print(hcx, hcy, cx,cy)

    cv2.imshow('Tracking', image)
    cv2.imshow('Clean', imgClean)
    cv2.waitKey(1)

    
