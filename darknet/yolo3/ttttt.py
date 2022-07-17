from itertools import count
from cv2 import destroyAllWindows
import keyboard
import os
import cv2
import imutils

# arr = os.listdir('C:\\Users\\Admin\\Desktop\\Newfolder\\biensoxechuso\\greenparkingchar\\greenparkingchar')
arr = os.listdir('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\x3')
for a in arr:
    img = cv2.imread('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\x3\\'+a)
    img2 = imutils.resize(img, width=100)
    print(a + '--' + str(len(os.listdir('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\x3'))))
    cv2.imshow(a, img2)
    
    b = cv2.waitKey(0) % 256
    if b == ord("0"):
        # countfile = len(os.listdir('data_char\\0'))
        cv2.imwrite('data_char\\0\\' + a, img) 
    elif b == ord("1"):
        # countfile = len(os.listdir('data_char\\1'))
        cv2.imwrite('data_char\\1\\' + a, img)
    elif b == ord("2"):
        # countfile = len(os.listdir('data_char\\2'))
        cv2.imwrite('data_char\\2\\' + a, img)
    elif b == ord("3"):
        # countfile = len(os.listdir('data_char\\3'))
        cv2.imwrite('data_char\\3\\' + a, img)
    elif b == ord("4"):
        # countfile = len(os.listdir('data_char\\4'))
        cv2.imwrite('data_char\\4\\' + a, img)
    elif b == ord("5"):
        # countfile = len(os.listdir('data_char\\5'))
        cv2.imwrite('data_char\\5\\' + a, img)
    elif b == ord("6"):
        # countfile = len(os.listdir('data_char\\6'))
        cv2.imwrite('data_char\\6\\' + a, img)
    elif b == ord("7"):
        # countfile = len(os.listdir('data_char\\7'))
        cv2.imwrite('data_char\\7\\' + a, img)
    elif b == ord("8"):
        # countfile = len(os.listdir('data_char\\8'))
        cv2.imwrite('data_char\\8\\' + a, img)
    elif b == ord("9"):
        # countfile = len(os.listdir('data_char\\9'))
        cv2.imwrite('data_char\\9\\' + a, img)
    elif b == ord("a"):
        # countfile = len(os.listdir('data_char\\a'))
        cv2.imwrite('data_char\\a\\' + a, img)
    elif b == ord("b"):
        # countfile = len(os.listdir('data_char\\b'))
        cv2.imwrite('data_char\\b\\' + a, img)
    elif b == ord("c"):
        # countfile = len(os.listdir('data_char\\c'))
        cv2.imwrite('data_char\\c\\' + a, img)
    elif b == ord("d"):
        # countfile = len(os.listdir('data_char\\d'))
        cv2.imwrite('data_char\\d\\' + a, img)
    elif b == ord("e"):
        # countfile = len(os.listdir('data_char\\e'))
        cv2.imwrite('data_char\\e\\' + a, img)
    elif b == ord("f"):
        # countfile = len(os.listdir('data_char\\f'))
        cv2.imwrite('data_char\\f\\' + a, img)
    elif b == ord("g"):
        # countfile = len(os.listdir('data_char\\g'))
        cv2.imwrite('data_char\\g\\' + a, img)
    elif b == ord("h"):
        # countfile = len(os.listdir('data_char\\h'))
        cv2.imwrite('data_char\\h\\' + a, img)
    elif b == ord("k"):
        # countfile = len(os.listdir('data_char\\k'))
        cv2.imwrite('data_char\\k\\' + a, img)
    elif b == ord("l"):
        # countfile = len(os.listdir('data_char\\l'))
        cv2.imwrite('data_char\\l\\' + a, img)
    elif b == ord("m"):
        # countfile = len(os.listdir('data_char\\m'))
        cv2.imwrite('data_char\\m\\' + a, img)
    elif b == ord("n"):
        # countfile = len(os.listdir('data_char\\n'))
        cv2.imwrite('data_char\\n\\' + a, img)
    elif b == ord("p"):
        # countfile = len(os.listdir('data_char\\p'))
        cv2.imwrite('data_char\\p\\' + a, img)
    elif b == ord("r"):
        # countfile = len(os.listdir('data_char\\r'))
        cv2.imwrite('data_char\\r\\' + a, img)
    elif b == ord("s"):
        # countfile = len(os.listdir('data_char\\s'))
        cv2.imwrite('data_char\\s\\' + a, img)
    elif b == ord("t"):
        # countfile = len(os.listdir('data_char\\t'))
        cv2.imwrite('data_char\\t\\' + a, img)
    elif b == ord("u"):
        # countfile = len(os.listdir('data_char\\u'))
        cv2.imwrite('data_char\\u\\' + a, img)
    elif b == ord("v"):
        # countfile = len(os.listdir('data_char\\v'))
        cv2.imwrite('data_char\\v\\' + a, img)
    elif b == ord("x"):
        # countfile = len(os.listdir('data_char\\x'))
        cv2.imwrite('data_char\\x\\' + a, img)
    elif b == ord("y"):
        # countfile = len(os.listdir('data_char\\y'))
        cv2.imwrite('data_char\\y\\' + a, img)
    elif b == ord("z"):
        # countfile = len(os.listdir('data_char\\z'))
        cv2.imwrite('data_char\\z\\' + a, img)
    elif b == ord("j"):
        # countfile = len(os.listdir('data_char\\z'))
        cv2.imwrite('data_char\\z1\\' + a, img)
    
    cv2.destroyAllWindows()
    try: 
        os.remove('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\yolo3\\data_char\\x3\\'+a)
    except: pass