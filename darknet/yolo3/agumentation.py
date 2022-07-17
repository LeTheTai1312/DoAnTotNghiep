import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cv2
import os
print(tf.__version__)                          
from tensorflow.keras import layers
from tensorflow import keras

def agu(path):
    arr = os.listdir('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\' + "b")
    try:
        os.makedirs("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\gen_"+ path, exist_ok = True)
        print("Directory '%s' created successfully")
    except OSError as error:
        print("Directory '%s' can not be created")
    n = 0
    for a in arr:
        n+=1
        print("count {} of {}".format(n, path))
        image = keras.preprocessing.image.load_img("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\" + "b" + "\\" + a)
        input_arr = keras.preprocessing.image.img_to_array(image)

        data = tf.expand_dims(input_arr, 0)
        myImageGen = keras.preprocessing.image.ImageDataGenerator(shear_range= 45)

        i= 0
        # os.mkdir("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\gen_"+ path)
        for batch in myImageGen.flow(data, batch_size=1, save_to_dir="C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\gen_"+ path, save_prefix='gen_' + path, save_format="jpg"):
            i+=1
            if i > 100:
                break

agu("b")




    
  







