import cv2
from pathlib import Path
import argparse
import time
import random

from numpy import imag

from lp_recognition import E2E
import os

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default="testtt/0000_00532_b.jpg")
    print(arg.parse_args())
    return arg.parse_args()

args = get_arguments()
img_path = Path(args.image_path)

# read image
img = cv2.imread("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt\\biensoxe84odau_optimized.jpg")

# start
# start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load model
model = E2E()
image = model.predict(img)
cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)
cv2.destroyAllWindows()
arr = os.listdir("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt")
# random.shuffle(arr)
# for a in arr:
#     start = time.time()
#     img = cv2.imread("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\testtt\\"+a)
#     image = model.predict(img)
#     end = time.time()
#     print('Model process on %.2f s' % (end - start))
#     cv2.imshow('License Plate', image)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         exit(0)
#     cv2.destroyAllWindows()
# recognize license plate
# image = model.predict(img)




# print('Model process on %.2f s' % (end - start))

# show image
# cv2.imshow('License Plate', image)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     exit(0)
# cv2.destroyAllWindows()
