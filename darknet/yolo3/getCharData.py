import imp
import cv2
import keyboard
import argparse
import time

from pathlib import Path
from lp_recognition import E2E
import os
from skimage.filters import threshold_local
from imutils import perspective
import imutils
from skimage import measure
import numpy as np
from data_utils import  convert2Square
from keras.models import load_model
import matplotlib.pyplot as plt




def get_arguments(str):
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default=str)
    # print(arg.parse_args())
    return arg.parse_args()

def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def get_candidate(path):
    candidates = []
    args = get_arguments(path)
    img_path = Path(args.image_path)
    # print(img_path)
    # read image
    img = cv2.imread(str(img_path))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = E2E()
    image = model.predict(img)
    image = imutils.resize(image, width=400)
    # cv2.imshow('image', imutils.resize(image, width=400))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     exit(0)
    # cv2.destroyAllWindows()

    image = np.array(255 * (image / 255) ** 2.0 , dtype='uint8')
   

    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    
    kernel = np.ones((3,3),np.uint8)
    V = imutils.resize(V, width=600)
    V = cv2.GaussianBlur(V ,(7,7),0)
    thresh1 = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 18)
    thresh1 = cv2.bitwise_not(thresh1)
    thresh = imutils.resize(thresh1, width=400)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.medianBlur(thresh, 7)


    labels = measure.label(thresh, connectivity=2, background=0)
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        print(mask)
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(contour)
            # if h > 100:
            #     print(x,y,w,h)
            # cv2.imshow('image', mask)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     exit(0)
            # cv2.destroyAllWindows()

            # rule to determine characters
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            heightRatio = h / float(mask.shape[0])
            widthRatio = w / float(mask.shape[0])
            # print(mask.shape)
            if 0.1 < aspectRatio < 1.0 and solidity > 0.25 and 0.25 < heightRatio < 0.4 :
                # extract characters
                # print(x,y,w,h,mask.shape)
                candidate = np.array(mask[y:y + h, x:x + w])
                square_candidate = convert2Square(candidate)
                square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                square_candidate = square_candidate.reshape((28, 28, 1))
                candidates.append((square_candidate, (y, x)))
                
    return candidates






arr = os.listdir('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt')
m = 0
model = load_model('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\weight\\modelcc2.h5')
classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A", 11: "C", 12: "D", 13: "E", 14: "F", 15: "G", 16: "H", 17: "K", 18: "L", 19: "M",
           20: "N", 21: "None", 22: "P", 23: "R", 24: "S", 25: "T", 26: "U", 27: "V", 28: "X", 29: "Y", 30: "Z"}
for a in arr:
    candidate = get_candidate('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt\\'+a)
    arr = []
    for i in candidate:
        img_array = i[0]/255
        plt.imshow(img_array, cmap = "gray")
        plt.show
        img_array = img_array.reshape(1,28,28,1)
        pred = np.argmax(model.predict(img_array))
        arr.append(classes[pred])
        
    print(arr)
    img = cv2.imread('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt\\'+a)
    cv2.imshow('image', imutils.resize(img, width=400))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit(0)
    cv2.destroyAllWindows()
