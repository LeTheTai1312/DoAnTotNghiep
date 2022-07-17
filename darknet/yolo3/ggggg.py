import imp
import os, shutil
import cv2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

arr = os.listdir("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen")
count = 0
for i in arr:
    arr2 = os.listdir("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\"+i)
    n = int(len(arr2)*0.15)
    count = 0
    try:
        os.makedirs("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data\\test\\"+ i, exist_ok = True)
        print("Directory '%s' created successfully", i)
    except OSError as error:
        print("Directory '%s' can not be created")
    for j in arr2:
        count += 1
        img = cv2.imread("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\"+i + "\\"+j)
        # cv2.imshow('image', img)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     exit(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data\\test\\'+i + "\\" +j, img)
        os.remove("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen\\"+i + "\\"+j)
        if count > n:
            break







# print(arr)
# train = ImageDataGenerator(rescale = 1/255)

# train_dataset = train.flow_from_directory("C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_gen",
#                                                 target_size= (28,28),
#                                                 batch_size = 3,
#                                                 class_mode = "categorical")

# print(train_dataset[0][1])