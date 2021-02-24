import numpy as np
import argparse
import cv2
import os
import csv
import pandas as pd


class ImageDetector:
    def __init__(
        self,
        weights,
        config_file,
        labels="yolo-files/coco.names",
        average_size="/yolo-files/average_size.csv",
        confidence=0.5,
        threshold=0.3,
    ):

        id_dictionary = {}
        with open(average_size, "r") as data:
            for line in csv.reader(data):
                element = line.pop(0)
                # remove -1 after correcting numbers in csv files ( should start from 0, not from 1)
                id_dictionary[int(element) - 1] = line

        self.average_size_dictionary = id_dictionary
        self.labels = open(labels).read().strip().split("\n")
        self.confidence = confidence
        self.threshold = threshold
        self.average_size = csv.DictReader(average_size)
        self.colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8"
        )
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights)
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [
            self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    """
     YOLO Image Detector. Allows or both object detection and distance to object measurer.
     In order to measure distance, you should load a file with the WIDTH of each possible object. 
     The focal length of the sensor is required for correct distance measurement. 
     The algoritm uses detected object width in pixels and calculates the distance base on true width
     and focal length: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
     
     Allows distance measurements using width or height.
    """

    def detect(self, image):
        boxes, confidences, classIDs, idxs = self.make_prediction(image)

        image, distances = self.draw_bounding_boxes(
            image, boxes, confidences, classIDs, idxs
        )

        return image, distances

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
                    centerX, centerY, w, h = box.astype("int")

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return boxes, confidences, classIDs

    def draw_bounding_boxes(self, image, boxes, confidences, classIDs, idxs):
        """Draws bounding boxes for detected objects.

        Allows for both width or heigh distance measurements.
        """

        distance_vector = {}

        if len(idxs) > 0:
            for i in idxs.flatten():

                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                d = self.get_distance(classIDs[i], w, h)
                distance_vector[d] = classIDs[i]

                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}: {:.2f}".format(
                    self.labels[classIDs[i]], confidences[i], d
                )
                cv2.putText(
                    image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        return image, distance_vector

    def make_prediction(self, image):

        height, width = image.shape[:2]

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = self.extract_boxes_confidences_classids(
            outputs, width, height
        )

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        return boxes, confidences, classIDs, idxs

    def get_distance(self, item_id, width, height, focal_length=596):
        mode = int(self.average_size_dictionary[item_id][0])
        value = self.average_size_dictionary[item_id][1]
        denominator = 0

        if mode == 0:
            denominator = height

        elif mode == 1:
            denominator = width

        elif mode == 2:
            max_val = max(width, height)
            denominator = max_val

        elif mode == 3:
            denominator = min(width, height)

        d = focal_length * float(value) / int(denominator)

        return d


def test():

    w = "yolo-files/yolov4-tiny.weights"
    cfg = "yolo-files/yolov4-tiny.cfg"
    l = "yolo-files/coco.names"
    avg = "yolo-files/average_size.csv"

    imgDet = ImageDetector(w, cfg, l, avg)

    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()

        if ret is True:
            detected, dictiz = imgDet.detect(frame)
            cv2.imshow("name", detected)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()

    cv2.destroyAllWindows()


# test()
