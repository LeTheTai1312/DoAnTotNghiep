# import time
# import cv2
# import numpy as np

# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#     return output_layers

# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
#     label = str(classes[class_id])
#     color = COLORS[class_id]
#     cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
#     cv2.putText(img, label + str(confidence) , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def savePredict(name, text):
#     textName = name + '.txt'
#     with open(textName, 'w+') as groundTruth:
#         groundTruth.write(text)
#         groundTruth.close()

# img_path = 'img.jpg'
# image = cv2.imread('C:\\Users\\Admin\\Desktop\\yolo\\darknet\\testImg\\0510_07155_b.jpg') # đổi tên ảnh để nhận dạng
# # cv2.imshow('image',image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# Width = image.shape[1]
# Height = image.shape[0]
# scale = 0.00392

# classes = None
# with open("yolo.names", 'r') as f: # Chỉnh sửa file yolo.names ở đây
#     classes = [line.strip() for line in f.readlines()]
#     COLORS = np.random.uniform(0, 255, size=(len(classes), 3)) 
#     net = cv2.dnn.readNet("yolov3-tiny_8000.weights", "yolov3-tiny.cfg") #Thay đổi tên của file weights và file cfg tại đây. 
#     blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False) #Convert ảnh sang blob
      
# net.setInput(blob)
# outs = net.forward(get_output_layers(net))

# class_ids = []
# confidences = []
# boxes = []
# conf_threshold = 0.5 #Đây là ngưỡng vật thể, nếu xác suất của vật thể nhỏ hơn 0.5 thì #model sẽ loại bỏ vật thể đó. Các bạn có thể tăng lên cao hoặc giảm xuống tùy theo mục #đích model của mình.
# nms_threshold = 0.4
# #Nếu có nhiều box chồng lên nhau, và vượt quá giá trị 0.4 (tổng diện tích chồng nhau) thì #1 trong 2 box sẽ bị loại bỏ.
# start = time.time() #đo thời gian thực thi của model

# for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.05:
#             print(confidence)
#             center_x = int(detection[0] * Width)
#             center_y = int(detection[1] * Height)
#             w = int(detection[2] * Width)
#             h = int(detection[3] * Height)
#             x = center_x - w / 2
#             y = center_y - h / 2
#             class_ids.append(class_id)
#             confidences.append(float(confidence))
#             boxes.append([x, y, w, h])

# indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# coordinates = []
# for i in indices:
#     index = i[0]
#     x_min, y_min, width, height = boxes[index]
#     x_min = round(x_min)
#     y_min = round(y_min)

#     coordinates.append((x_min, y_min, width, height))
# # Result = ""
# # a = {}
# # for i in indices:
# #     i = i[0]
# #     box = boxes[i]
# #     x = box[0]
# #     a[0] = box[0]
# #     y = box[1]
# #     a[1] = box[1]
# #     w = box[2]
# #     a[2] = box[2]
# #     h = box[3]
# #     a[3] = box[3]
# #     a[4] = str(class_ids[i])
# #     a[5] = confidences[i] 
# #     textpredict = "{} {} {} {} {} {}\n".format(str(class_ids[i]), confidences[i], x, y, x+w, y+h)
# #     Result += textpredict
# # savePredict('name', Result)

# # scale_percent = 50
# # width = int(image.shape[1] * scale_percent / 100)
# # height = int(image.shape[0] * scale_percent / 100)
# # image = cv2.resize(src=image, dsize=(width,height))
# # draw_prediction(image, 0, a[5],  a[0], a[1], a[2], a[3] )
# # color = color = (0, 255, 0)
# # cv2.rectangle(image, (int(a[0]), int(a[1])), (int(a[0])+int(a[2]), int(a[1])+int(a[3])), color, 1)
# # cv2.rectangle(image, (b[0], b[1]), (b[0] + 50, b[1] + 50), color, 1)
# # cv2.putText(image, 'lp' + str(a[5]) , (int(a[0] - 10), int(a[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
# # cv2.imshow("object detection", image)

# # end = time.time()
# # def order_point(pts):
# #     rect = np.zeros((4, 2), dtype = "float32")
# #     s = pts.sum(axis = 1)
# # 	rect[0] = pts[np.argmin(s)]
# # 	rect[2] = pts[np.argmax(s)]
    
# # print("YOLO Execution time: " + str(end-start))
# # cv2.waitKey()