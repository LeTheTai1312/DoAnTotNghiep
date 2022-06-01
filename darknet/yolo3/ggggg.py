from itertools import count
from cv2 import destroyAllWindows
import keyboard
import os
import cv2
import imutils
from data_utils import convert2Square

img = cv2.imread('C:\\Users\\Admin\\Desktop\\Newfolder\\biensoxechuso\\charTrainset\\charTrainset\\5\\46409_3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 12)
square_candidate = convert2Square(img)
square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
square_candidate = square_candidate.reshape((28, 28, 1))

cv2.imshow('ggg.jpg', square_candidate)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)
cv2.destroyAllWindows()
cv2.imwrite('ggg.jpg', square_candidate)
