import uuid
import time

import numpy as np
import argparse
import cv2
import os, sys
import csv
# import pandas as pd
# import tensorflow as tf
from shapely.geometry import Polygon
from darknet import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from storage import DetectedPipeline
from storage import logger, timer, get_poly_lines

# from yolov3.utils import (
#     detect_image,
#     detect_realtime,
#     detect_video,
#     Load_Yolo_model,
#     detect_video_realtime_mp,
# )
# from yolov3.configs import *


class DetectedObject:
    def __init__(self, id, score, bbox, label, color):
        self.id = id
        self.score = score
        self.bbx = bbox
        self.color = color
        self.label = label
        self.unique_id = uuid.uuid4()


class ImageDetector:
    """
    YOLO Image Detector. Allows or both object detection and distance to object measurer.
    In order to measure distance, you should load a file with the WIDTH of each possible object.
    The focal length of the sensor is required for correct distance measurement.
    The algoritm uses detected object width in pixels and calculates the distance base on true width
    and focal length: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

    Allows distance measurements using width or height.
    """

    def __init__(
        self,
        weights,
        config_file,
        mode="CPU",
        labels="yolo-files/coco.names",
        data_file="yolo-files/coco.data",
        average_size="yolo-files/average_size.csv",
        confidence=0.5,
        threshold=0.3,
    ):

        # for CPU detection
        if mode == "CPU":
            self.mode = 0
            log_text("CPU mode")

            # try to use with cudnn backend
            self.net = cv2.dnn.readNetFromDarknet(config_file, weights)
            self.layer_names = self.net.getLayerNames()
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except:
                print("No GPU found")
            self.layer_names = [
                self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
            ]

        # for GPU detection
        elif mode == "GPU":
            log_text("GPU mode")
            self.mode = 1
            self.network, self.class_names, class_colors = load_network(config_file, data_file, weights)
            self.network_width = network_width(self.network)
            self.network_height = network_height(self.network)
            
        self.det_time = 0
        self.pre_time = 0
        self.pro_time = 0

        self.labels = open(labels).read().strip().split("\n")

        # start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # log_file_name = "{:s}_{:s}.log".format("LOG_FILE", start_time)
        # log_file_path = os.path.join(os.getcwd(), log_file_name)

        # logger.add("log.file", level="DEBUG", format="{time}{level}{message}")

        id_dictionary = {}
        with open(average_size, "r") as data:
            for line in csv.reader(data):
                element = line.pop(0)
                # remove -1 after correcting numbers in csv files ( should start from 0, not from 1)
                id_dictionary[int(element) - 1] = line
        log_text("Dictionary read!")
        self.average_size_dictionary = id_dictionary

        self.confidence = confidence
        self.threshold = threshold
        self.start = time.time()
        self.average_size = csv.DictReader(average_size)
        self.colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8"
        )


    def _get_distances(self, mode, detected_list, height, width):
        """ Calculates distances and checks if vehicles in front.
        Returns two dictionaries. One for distances having key:distance, and value:object_id
        and another one for objects that were detected in front having key:object_id and value:0"""

        # TO DO:
        # improve for semapthore distance estimation
        # extend range for pedestrians

        distance_vector = {}
        frontal_objects = {}

        # polygon used for lane detection and collision system
        p1 = Polygon(get_poly_lines("poly"))
     
        if detected_list:
            for detected_object in detected_list:
                x, y = detected_object.bbx[0], detected_object.bbx[1]
                w, h = detected_object.bbx[2], detected_object.bbx[3]

                # calculate overlap area
                object_area = w * h
                object_area_min_val = 80 / 100 * object_area
                y_cart = height - y - h

                p2 = Polygon(
                    [(x, y_cart), (x + w, y_cart), (x + w, y_cart + h), (x, y_cart + h)]
                )

                # if the car is in front and it is over 80% inside the riding view
                # calculate distance and add as possible danger

                intersection = p2.intersects(p1)

                if intersection:
                    intersection_area = p2.intersection(p1).area

                    if intersection_area >= object_area_min_val:
                        distance = self._get_distance(detected_object.id, w, h)
                        distance_vector[distance] = detected_object.id

                        id = detected_object.id
                        frontal_objects[id] = 0

        return distance_vector, frontal_objects

    def detect(self, image) -> DetectedPipeline:
        """
        Public method used as entry point for image detection.

        Returns a DetectedPipeline object.
        """

        # LOAD IMAGE SHAPE FROM CONFIG/ FIRST MESSAGE
        height = image.shape[0]
        width = image.shape[1]

        detected_obj = DetectedPipeline(image)
        detected_list = self._make_prediction(image)

        if detected_list:
            distances, frontal_list = self._get_distances(
                True, detected_list, height, width
            )
            detected_obj.frontal_distances = distances
            detected_obj.frontal_objects = frontal_list
            detected_obj.detected_objects = detected_list
            detected_obj.detected = True

        self.start = time.time()
        return detected_obj

    def _extract_boxes_confidences_classids(self, outputs, width, height):
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

    def _make_prediction(self, image) -> [DetectedObject]:
        """
        Private method which passes the image through the CNN.
        Extracts class confidences and objects boxes.
        Applies NMS Suppression.

        Returns list of DetectedObjects.
        """

        height, width = image.shape[:2]
        detected_objects = []

        # CPU detection
        if self.mode == 0:
            # Create a blob and pass it through the model
            blob = cv2.dnn.blobFromImage(
                image, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward(self.layer_names)

            # Extract bounding boxes, confidences and classIDs
            boxes, confidences, classIDs = self._extract_boxes_confidences_classids(
                outputs, width, height
            )

            # Apply Non-Max Suppression
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    label = self.labels[classIDs[i]]
                    color = [int(c) for c in self.colors[classIDs[i]]]

                    detected_object = DetectedObject(
                        classIDs[i], confidences[i], boxes[i], label, color
                    )
                    detected_objects.append(detected_object)

        # GPU detection
        else:
            start_time = time.time()
            detections, width_ratio, height_ratio = self._detect_gpu(image)
            end_time = time.time()


            for label, conf, bboxes in detections:
                boxes = []

                class_id = int(self.labels.index(label))
                confidence = float(conf)

                if confidence >= self.confidence:
                    box = bboxes * np.array([width_ratio, height_ratio, width_ratio, height_ratio])
                    centerX, centerY, w, h = box.astype("int")

                    x = int(centerX - (w/2))
                    y = int(centerY - (h/2))

                    boxes.append(x)
                    boxes.append(y)
                    boxes.append(int(w))
                    boxes.append(int(h))

                    color = [int(c) for c in self.colors[class_id]]

                    detected_object = DetectedObject(
                        class_id, confidence, boxes, label, color
                    )
                    detected_objects.append(detected_object)

            final_end_time = time.time()

        return detected_objects
    
    def _detect_gpu(self, image):

        darknet_image = make_image(self.network_width, self.network_height, 3)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.network_width, self.network_height),
                            interpolation=cv2.INTER_LINEAR)
        
        # get image ratios to conver bbx to proper size

        image_height, image_width, _ = image.shape
        width_ratio = image_width/self.network_width
        height_ratio = image_height/self.network_height

        # run model on darknet style image to get detections
        copy_image_from_bytes(darknet_image, img_resized.tobytes())
        detections = detect_image(self.network, self.class_names, darknet_image)
        free_image(darknet_image)

        return detections, width_ratio, height_ratio

    def _get_distance(self, item_id, width, height, focal_length=596) -> float:
        """
        Calculates distance to the object based on the dictionary configuration.

        Returns the distance in cm as float.
        """

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

        return d / 100  # convert from m to cm



def log_text(text):
    logger.log("IMAGE_DETECTOR", text)


def test():

    w = "yolo-files/yolov4-tiny.weights"
    cfg = "yolo-files/yolov4-tiny.cfg"
    l = "yolo-files/coco.names"
    avg = "yolo-files/average_size.csv"
    image = "yolo-files/test3.png"

    imgDet = ImageDetector(
        "yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg"
    )
    img = cv2.imread(image)

    start = time.time()
    dictiz = imgDet.detect(img)
    end = time.time()
    detected_objs = dictiz.detected_objects

    print("Total obj detected: ", len(detected_objs))
    print("Total time: ", end - start)


def test_gpu():
    imageDet = ImageDetector(
        mode="GPU",
        weights="yolo-files/yolov4-tiny.weights",
        config_file="yolo-files/yolov4-tiny.cfg",
    )

    image = cv2.imread("images/kite.jpg")
    print("Image shape: ", image.shape)
    start_time = time.time()
    detected = imageDet.detect(image)
    end_time = time.time()

    for i in detected.detected_objects:
        print("--- new object ---")
        print("Class: ", i.id)
        print("Score: ", i.score)
        print("Label: ", i.label)

    print("Detected objects on GPU: ", detected)
    print("Total detected time: ", end_time - start_time)


# test_gpu()
#test()
