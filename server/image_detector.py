import numpy as np
import argparse
import cv2
import os
import csv
import pandas as pd

from shapely.geometry import Polygon
# from pycoral.adapters.common import input_size
# from pycoral.adapters.detect import get_objects
# from pycoral.utils.dataset import read_label_file
# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.utils.edgetpu import run_inference

import uuid
import time

from storage import DetectedPipeline
from storage import get_polygone


class DetectedObject:
    def __init__(self, id, score, bbox, label, color):
        self.id = id
        self.score = score
        self.bbx = bbox
        self.color = color
        self.label = label
        self.unique_id = uuid.uuid4()


class ImageDetector:
    def __init__(
        self,
        weights,
        config_file,
        labels="yolo-files/coco.names",
        average_size="yolo-files/average_size.csv",
        confidence=0.5,
        threshold=0.3,
    ):


        self.mode = 0
        self.net = cv2.dnn.readNetFromDarknet(config_file, weights)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            print("No GPU found")

        self.layer_names = self.net.getLayerNames()
        self.layer_names = [
            self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        self.labels = open(labels).read().strip().split("\n")

        id_dictionary = {}
        with open(average_size, "r") as data:
            for line in csv.reader(data):
                element = line.pop(0)
                # remove -1 after correcting numbers in csv files ( should start from 0, not from 1)
                id_dictionary[int(element) - 1] = line

        self.average_size_dictionary = id_dictionary

        self.confidence = confidence
        self.threshold = threshold
        self.start = time.time()
        self.average_size = csv.DictReader(average_size)
        self.colors = np.random.randint(
            0, 255, size=(len(self.labels), 3), dtype="uint8"
        )

    # """
    #  YOLO Image Detector. Allows or both object detection and distance to object measurer.
    #  In order to measure distance, you should load a file with the WIDTH of each possible object.
    #  The focal length of the sensor is required for correct distance measurement.
    #  The algoritm uses detected object width in pixels and calculates the distance base on true width
    #  and focal length: https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

    #  Allows distance measurements using width or height.
    # """

    def __get_distances(self, mode, detected_list, height, width):
        # """ Calculates distances and checks if vehicles in front"""

        # Done:
        # check distance only for cars in front

        # TO DO:
        # improve for semapthore distance estimation
        # extend range for pedestrians

        distance_vector = {}
        frontal_objects = {}

        # polygon used for lane detection
        # p1 = Polygon([(340, 150), (920, height - 550), (1570, 150)])

        p1 = Polygon(get_polygone("poly"))

        if detected_list:
            for i in detected_list:
                x, y = i.bbx[0], i.bbx[1]
                w, h = i.bbx[2], i.bbx[3]
                # print("X: " + str(x) + " Y: " +str(y) + " W: " + str(w) + " H: " + str(h))
                # calculate overlap area
                car_area = w * h
                car_area_min_val = 80 / 100 * car_area
                y_cart = height - y - h

                p2 = Polygon(
                    [(x, y_cart), (x + w, y_cart), (x + w, y_cart + h), (x, y_cart + h)]
                )
                # print("Second polygone coordinates: ")
                # print(x, y_cart)
                # print(x+w, y_cart)
                # print(x+w, y_cart +h)
                # print(x, y_cart +h)

                # if the car is in front and it is over 80% inside the riding view
                # calculate distance and add as possible danger

                intersection = p2.intersects(p1)

                if intersection:
                    intersection_area = p2.intersection(p1).area
                    #print("Overlap procent: ", intersection_area / car_area)
                    if intersection_area >= car_area_min_val:
                        #print("more than 80% in the interesting zone")
                        d = self.get_distance(i.id, w, h)
                        distance_vector[d] = i.id
                        id = i.unique_id
                        frontal_objects[id] = 0

        return distance_vector, frontal_objects

    def detect(self, image):

        height = image.shape[0]
        width = image.shape[1]

        # if time.time() - self.start > 0.1:
        detected_obj = DetectedPipeline(image)

        if self.mode == 0:  # yolo modeld
            detected_list = self.make_prediction(image)

            if detected_list:
                distances, frontal_list = self.__get_distances(
                    True, detected_list, height, width
                )
                detected_obj.frontal_distances = distances
                detected_obj.frontal_objects = frontal_list
                detected_obj.detected_objects = detected_list
                detected_obj.detected = True

            self.start = time.time()
            return detected_obj

        else:
            cv2_im = image
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
            run_inference(self.interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(self.interpreter, self.confidence)[: self.top_k]

            detected_list = []
            height, width, channels = cv2_im.shape
            scale_x, scale_y = (
                width / self.inference_size[0],
                height / self.inference_size[1],
            )
            # convert from object to detectedObject
            for obj in objs:
                bbox = obj.bbox.scale(scale_x, scale_y)
                x0, y0 = int(bbox.xmin), int(bbox.ymin)
                x1, y1 = int(bbox.xmax) - x0, int(bbox.ymax) - y0
                bbox = [x0, y0, x1, y1]

                label = self.labels[obj.id]
                color = self.colors[obj.id]
                detected_obj = DetectedObject(obj.id, obj.score, bbox, label, color)
                detected_list.append(detected_obj)

            if detected_list:
                distances, frontal_list = self.__get_distances(
                    True, detected_list, height
                )
                draw_box_parameters = (detected_list, distances, frontal_list)
            else:
                draw_box_parameters = ()

            self.start = time.time()
            return draw_box_parameters
        # else:
        #     return ()
            # cv2_im, dictionary = self.append_objs_to_img(cv2_im, self.inference_size, objs, self.labels)

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

    def append_objs_to_img(self, cv2_im, inference_size, objs, labels):

        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        distance_vector = {}

        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)

            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            index = labels.get(obj.id, obj.id)

            percent = int(100 * obj.score)
            label = "{}% {}".format(percent, self.labels.get(obj.id, obj.id))

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(
                cv2_im,
                label,
                (x0, y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        return cv2_im, distance_vector

    # def draw_bounding_boxes(self, image, boxes, confidences, classIDs, idxs):
    #     """Draws bounding boxes for detected objects.

    #     Allows for both width or heigh distance measurements.
    #     """
    #     distance_vector = {}

    #     if len(idxs) > 0:
    #         for i in idxs.flatten():

    #             x, y = boxes[i][0], boxes[i][1]
    #             w, h = boxes[i][2], boxes[i][3]

    #             d = self.get_distance(classIDs[i], w, h)
    #             distance_vector[d] = classIDs[i]

    #             color = [int(c) for c in self.colors[classIDs[i]]]
    #             cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    #             text = "{}: {:.4f}: {:.2f}".format(
    #                 self.labels[classIDs[i]], confidences[i], d
    #             )
    #             cv2.putText(
    #                 image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    #             )

    #     return image, distance_vector

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
        detected_objects = []

        if len(idxs) > 0:
            for i in idxs.flatten():

                label = self.labels[classIDs[i]]
                # print("detected: ", label)
                color = [int(c) for c in self.colors[classIDs[i]]]
                detected_object = DetectedObject(
                    classIDs[i], confidences[i], boxes[i], label, color
                )
                detected_objects.append(detected_object)

        return detected_objects
        # return boxes, confidences, classIDs, idxs, uniqueIDs

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

        return d / 100  # convert from m to cm


def test():

    w = "yolo-files/yolov4-tiny.weights"
    cfg = "yolo-files/yolov4-tiny.cfg"
    l = "yolo-files/coco.names"
    avg = "yolo-files/average_size.csv"
    image = "yolo-files/test3.png"

    imgDet = ImageDetector("yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg")
    img = cv2.imread(image)

    start = time.time()
    dictiz = imgDet.detect(img)
    end = time.time()
    detected_objs = dictiz.detected_objects
    
    
    print("Total obj detected: ", len(detected_objs))
    print("Total time: ", end-start)



# test()
