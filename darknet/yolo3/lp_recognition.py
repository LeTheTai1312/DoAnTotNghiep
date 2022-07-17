import imp
import time
import cv2
import numpy as np
from lp_detection.detect import detectNumberPlate
from data_utils import order_points, draw_labels_and_boxes
from imutils import perspective
import imutils
from skimage import measure
from data_utils import convert2Square
from keras.models import load_model

LP_DETECTION_CFG = {
    "weight_path": "weight\yolov3-tiny_8000.weights",
    "classes_path": "lp_detection\cfg\yolo.names",
    "config_path": "lp_detection\cfg\yolov3-tiny.cfg"
}

CLASSES = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A", 11: "C", 12: "D", 13: "E", 14: "F", 15: "G", 16: "H", 17: "K", 18: "L", 19: "M",
           20: "N", 21: "None", 22: "P", 23: "R", 24: "S", 25: "T", 26: "U", 27: "V", 28: "X", 29: "Y", 30: "Z"}


CHAR_CLASSIFICATION_WEIGHTS = 'C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\weight\\modelcc2.h5'


class E2E(object):
    def __init__(self):
        self.image = np.empty((416, 416, 3))
        self.detectLP = detectNumberPlate(
            LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        self.recogChar = load_model(CHAR_CLASSIFICATION_WEIGHTS)
        # self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.candidates = []

    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            ValueError('No images detected')

        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        # Input image or frame
        self.image = image
        lp = "oh_my_god"
        for coordinate in self.extractLP():     # detect license plate by yolov3
            self.candidates = []

            # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # crop number plate used by bird's eyes view transformation
            LpRegion = perspective.four_point_transform(self.image, pts)

            self.segmentation(LpRegion)

            self.recognizeChar()

            license_plate = self.format()

            lp = license_plate

            # self.image = LpRegion

            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image
        

    def segmentation(self, img):

        image = imutils.resize(img, width=400)

        image = np.array(255 * (image / 255) ** 1.5, dtype='uint8')

        V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

        kernel = np.ones((3, 3), np.uint8)
        V = imutils.resize(V, width=600)
        V = cv2.GaussianBlur(V, (7, 7), 0)
        thresh1 = cv2.adaptiveThreshold(
            V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 18)
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
            _, contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(mask.shape[0])
                widthRatio = w / float(mask.shape[0])
                # print(mask.size)
                if 0.1 < aspectRatio < 1.0 and solidity > 0.25 and 0.25 < heightRatio < 0.4:
                    # extract characters
                    # print(x,y,w,h,mask.shape)
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []
        for char, coordinate in self.candidates:
            # char = char/255
            characters.append(char)
            coordinates.append(coordinate)
            # print(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 21:    # if is background or noise, ignore it
                continue
            self.candidates.append((CLASSES[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []
        second_line = []
        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                # print(" {} - {} ". format(self.candidates[0][1][0], coordinate[0]))
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if(len(first_line) == 4):
            if(first_line[2][0] == "8"):
                first_line[2] = list(first_line[2])
                first_line[2][0] = "B"
                tuple(first_line[2])
            elif(first_line[2][0] == "0"):
                first_line[2] = list(first_line[2])
                first_line[2][0] = "D"
                tuple(first_line[2])
            elif(first_line[2][0] == "5"):
                first_line[2] = list(first_line[2])
                first_line[2][0] = "S"
                tuple(first_line[2])
            elif(first_line[2][0] == "2"):
                first_line[2] = list(first_line[2])
                first_line[2][0] = "Z"
                tuple(first_line[2])
            elif(first_line[2][0] == "1"):
                first_line[2] = list(first_line[2])
                first_line[2][0] = "T"
                tuple(first_line[2])
            
            
            elif(first_line[0][0] == "D"):
                first_line[0] = list(first_line[0])
                first_line[0][0] = "0"
                tuple(first_line[0])
            elif(first_line[0][0] == "S"):
                first_line[0] = list(first_line[0])
                first_line[0][0] = "5"
                tuple(first_line[0])
            elif(first_line[0][0] == "Z"):
                first_line[0] = list(first_line[0])
                first_line[0][0] = "2"
                tuple(first_line[0])
            elif(first_line[0][0] == "T"):
                first_line[0] = list(first_line[0])
                first_line[0][0] = "1"
                tuple(first_line[0])

            elif(first_line[1][0] == "D"):
                first_line[1] = list(first_line[1])
                first_line[1][0] = "0"
                tuple(first_line[1])
            elif(first_line[1][0] == "S"):
                first_line[1] = list(first_line[1])
                first_line[1][0] = "5"
                tuple(first_line[1])
            elif(first_line[1][0] == "Z"):
                first_line[1] = list(first_line[1])
                first_line[1][0] = "2"
                tuple(first_line[1])
            elif(first_line[1][0] == "T"):
                first_line[1] = list(first_line[1])
                first_line[1][0] = "1"
                tuple(first_line[1])

            elif(first_line[3][0] == "D"):
                first_line[3] = list(first_line[3])
                first_line[3][0] = "0"
                tuple(first_line[3])
            elif(first_line[3][0] == "S"):
                first_line[3] = list(first_line[3])
                first_line[3][0] = "5"
                tuple(first_line[3])
            elif(first_line[3][0] == "Z"):
                first_line[3] = list(first_line[3])
                first_line[3][0] = "2"
                tuple(first_line[1])
            elif(first_line[3][0] == "T"):
                first_line[3] = list(first_line[3])
                first_line[3][0] = "1"
                tuple(first_line[3])
                
        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + \
                "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate
