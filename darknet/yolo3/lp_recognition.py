import imp
import time
import cv2
import numpy as np
from lp_detection.detect import detectNumberPlate
from data_utils import order_points
from imutils import perspective

LP_DETECTION_CFG = {
    "weight_path": "weight\yolov3-tiny_8000.weights",
    "classes_path": "lp_detection\cfg\yolo.names",
    "config_path": "lp_detection\cfg\yolov3-tiny.cfg"
}

class E2E(object):
    def __init__(self):
        self.image = np.empty((416, 416, 3))
        self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        # self.recogChar = CNN_Model(trainable=False).model
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
           
            self.image = LpRegion

        # return self.image
        return self.image





