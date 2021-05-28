import cv2
import numpy as np
from glob import glob
import os

# Load Yolo
net = cv2.dnn.readNet(os.path.join(os.path.dirname(
    __file__), "yolov4.weights"), os.path.join(os.path.dirname(__file__), "yolov4.cfg"))
classes = []
with open(os.path.join(os.path.dirname(__file__), "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = (0, 255, 255)
DEFAULT_CONFIDENCE = 0.1
DEFAULT_IOU = 0.1

ipcam = 'cricket.mp4'

source = 0 if not ipcam else ipcam

readVideo = cv2.VideoCapture(source)

while True:
    # img = cv2.imread("ball.jpg", 1)
    status, frame = readVideo.read()
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    print(True)
    if status:

        img = cv2.resize(frame, (800,600))
        height, width, channels = img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > DEFAULT_CONFIDENCE:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        count = 0
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == "sports ball" or label == "apple":
                    try:
                        color = colors
                    except:
                        color = (0, 255, 100)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, 'Ball', (x, y + 30), font, 1, color, 1)
                    count = count+1
                else:
                    
                    pass

        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
    
    else:
        pass

cv2.destroyAllWindows()