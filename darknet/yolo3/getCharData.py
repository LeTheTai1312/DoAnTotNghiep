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
    # image = adjust_gamma(image, 0.2)
    # cv2.imshow('image', imutils.resize(image, width=400))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     exit(0)
    # cv2.destroyAllWindows()

    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    # V = cv2.GaussianBlur(V ,(7,7),0)
    # T = threshold_local(V, 9, offset=2, method="gaussian")
    # thresh1 = (V > T).astype("uint8") * 255
    kernel = np.ones((3,3),np.uint8)
    V = imutils.resize(V, width=600)
    V = cv2.GaussianBlur(V ,(7,7),0)
    thresh1 = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 18)
    thresh1 = cv2.bitwise_not(thresh1)
    thresh = imutils.resize(thresh1, width=400)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.medianBlur(thresh, 7)


    # cv2.imshow('image', thresh)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     exit(0)
    # cv2.destroyAllWindows()

    labels = measure.label(thresh, connectivity=2, background=0)
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

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
for a in arr:
    candidate = get_candidate('testtt/'+a)
    n = 0
    for i in candidate:
        cv2.imwrite('data_char\\x3\\' + str(m) +'_' + str(n) +'.jpg', i[0])
        n = n+1
    m = m + 1
#         cv2.imshow('image', i[0])
#         # 
#         a = cv2.waitKey(0) % 256
#         if a == ord("0"):
#             countfile = len(os.listdir('data_char\\0'))
#             cv2.imwrite('data_char\\0\\0_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("1"):
#             countfile = len(os.listdir('data_char\\1'))
#             cv2.imwrite('data_char\\1\\1_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("2"):
#             countfile = len(os.listdir('data_char\\2'))
#             cv2.imwrite('data_char\\2\\2_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("3"):
#             countfile = len(os.listdir('data_char\\3'))
#             cv2.imwrite('data_char\\3\\3_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("4"):
#             countfile = len(os.listdir('data_char\\4'))
#             cv2.imwrite('data_char\\4\\4_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("5"):
#             countfile = len(os.listdir('data_char\\5'))
#             cv2.imwrite('data_char\\5\\5_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("6"):
#             countfile = len(os.listdir('data_char\\6'))
#             cv2.imwrite('data_char\\6\\6_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("7"):
#             countfile = len(os.listdir('data_char\\7'))
#             cv2.imwrite('data_char\\7\\7_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("8"):
#             countfile = len(os.listdir('data_char\\8'))
#             cv2.imwrite('data_char\\8\\8_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("9"):
#             countfile = len(os.listdir('data_char\\9'))
#             cv2.imwrite('data_char\\9\\9_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("a"):
#             countfile = len(os.listdir('data_char\\a'))
#             cv2.imwrite('data_char\\a\\a_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("b"):
#             countfile = len(os.listdir('data_char\\b'))
#             cv2.imwrite('data_char\\b\\b_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("c"):
#             countfile = len(os.listdir('data_char\\c'))
#             cv2.imwrite('data_char\\c\\c_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("d"):
#             countfile = len(os.listdir('data_char\\d'))
#             cv2.imwrite('data_char\\d\\d_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("e"):
#             countfile = len(os.listdir('data_char\\e'))
#             cv2.imwrite('data_char\\e\\e_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("f"):
#             countfile = len(os.listdir('data_char\\f'))
#             cv2.imwrite('data_char\\f\\f_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("g"):
#             countfile = len(os.listdir('data_char\\g'))
#             cv2.imwrite('data_char\\g\\g_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("h"):
#             countfile = len(os.listdir('data_char\\h'))
#             cv2.imwrite('data_char\\h\\h_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("k"):
#             countfile = len(os.listdir('data_char\\k'))
#             cv2.imwrite('data_char\\k\\k_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("l"):
#             countfile = len(os.listdir('data_char\\l'))
#             cv2.imwrite('data_char\\l\\l_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("m"):
#             countfile = len(os.listdir('data_char\\m'))
#             cv2.imwrite('data_char\\m\\m_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("n"):
#             countfile = len(os.listdir('data_char\\n'))
#             cv2.imwrite('data_char\\n\\n_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("p"):
#             countfile = len(os.listdir('data_char\\p'))
#             cv2.imwrite('data_char\\p\\p_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("r"):
#             countfile = len(os.listdir('data_char\\r'))
#             cv2.imwrite('data_char\\r\\r_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("s"):
#             countfile = len(os.listdir('data_char\\s'))
#             cv2.imwrite('data_char\\s\\s_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("t"):
#             countfile = len(os.listdir('data_char\\t'))
#             cv2.imwrite('data_char\\t\\t_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("u"):
#             countfile = len(os.listdir('data_char\\u'))
#             cv2.imwrite('data_char\\u\\u_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("v"):
#             countfile = len(os.listdir('data_char\\v'))
#             cv2.imwrite('data_char\\v\\v_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("x"):
#             countfile = len(os.listdir('data_char\\x'))
#             cv2.imwrite('data_char\\x\\x_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("y"):
#             countfile = len(os.listdir('data_char\\y'))
#             cv2.imwrite('data_char\\y\\y_'+ str(countfile) +'.jpg', i[0])
#         elif a == ord("z"):
#             countfile = len(os.listdir('data_char\\z'))
#             cv2.imwrite('data_char\\z\\z_'+ str(countfile) +'.jpg', i[0])
#         cv2.destroyAllWindows()