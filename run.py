
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# This run currently have a model trained upon 3 classes A, B, C

cap = cv2.VideoCapture(0) # cv2 video capture object
detector = HandDetector(maxHands=1) # Limiting the number of hands to 1 for simplicity

# ⬇️ are the paths to the model and the labels file
modelKerasPath = "/Users/divyardhansinh/Documents/SEM 7/ML/project/SLR-master/Model/keras_model.h5"
modelLabelsPath = "/Users/divyardhansinh/Documents/SEM 7/ML/project/SLR-master/Model/labels.txt"
classifier = Classifier(modelKerasPath , modelLabelsPath) # This is our classifier Model Object

offset = 20
imgSize = 300

# folder = "Data/C"
counter = 0

labels = ["A", "B", "C" , "D" , "E" , "F" , "G" , "H" , "I" , "K" , "L" , "M" , "N" , "O" , "P" , "Q" , "R" ,  "T" , "U" , "V" , "W" , "X" , "Y" , "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
        except: # Here I would love to add the exception for the case when the empty resize values are passed to the view
            ...
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)