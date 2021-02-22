import numpy as np
import argparse
import cv2
import os

class ImageDetector:

    def __init__(self, weights, config_file, labels="yolo-files/coco.names", average_size="yolo-files/average_size", confidence=0.5, threshold=0.3):
        self.labels = open(labels).read().strip().split('\n')
        self.confidence = confidence
        self.threshold = threshold
        self.average_size = open(average_size).read().strip().split('\n')
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        

    def detect(self, image):
        boxes, confidences, classIDs, idxs = self.make_prediction( image)
        #return (image, boxes, confidences, classIDs, idxs)
        image = self.draw_bounding_boxes(image, boxes, confidences, classIDs, idxs)

        return image

    def extract_boxes_confidences_classids(self, outputs, width, height):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:            
                # Extract the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                
                # Consider only the predictions that are above the confidence threshold
                if conf > self.confidence:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype('int')

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)
        
        return boxes, confidences, classIDs

    def draw_bounding_boxes(self, image, boxes, confidences, classIDs, idxs):
        if len(idxs) > 0:
            for i in idxs.flatten():
                # if classIDs[i] == 41:
                    # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                #print("X este: ", x, " y este: ", y, "w este: ", w, " h este: ", h)
                self.get_distance(classIDs[i], w)
                
                #d = (focal_length*8)/h
                #print("Distance is: ", d)
                # draw the bounding box and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #cv2.putText(image, str(d) + " cm", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)

        return image

    def make_prediction(self, image):

        height, width = image.shape[:2]
        
        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = self.extract_boxes_confidences_classids(outputs, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
    
        return boxes, confidences, classIDs, idxs

    def get_distance(self, item_id, width, focal_length = 596):
        
        if item_id == 67:
            d = focal_length*7.4/width
            print("distance is: ", d)
        


def test():

    w = "yolov4-tiny.weights"
    cfg = "yolov4-tiny.cfg"
    l = "coco.names"
    avg = "average_size"

    imgDet = ImageDector(w,cfg,l, avg)

    vid = cv2.VideoCapture(0)

    while  True:
        ret, frame = vid.read()

        if ret is True:
            detected = imgDet.detect(frame)
            cv2.imshow('name', detected)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break    

    vid.release() 

    cv2.destroyAllWindows() 

