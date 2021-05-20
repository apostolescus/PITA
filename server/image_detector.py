import numpy as np
import argparse
import cv2
import os
import csv
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import loguru
from shapely.geometry import Polygon

import uuid
import time

from storage import DetectedPipeline
from storage import get_polygone

from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

class DetectedObject:
    def __init__(self, id, score, bbox, label, color):
        self.id = id
        self.score = score
        self.bbx = bbox
        self.color = color
        self.label = label
        self.unique_id = uuid.uuid4()

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

class ImageDetector:
    def __init__(
        self,
        weights,
        config_file,
        mode = "CPU",
        labels="yolo-files/coco.names",
        average_size="yolo-files/average_size.csv",
        confidence=0.5,
        threshold=0.3,
    ):
        # for CPU detection
        if mode == "CPU":
            self.mode = 0
            print("CPU mode")
            self.net = cv2.dnn.readNetFromDarknet(config_file, weights)
            self.layer_names = self.net.getLayerNames()
            self.layer_names = [
                self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
            ]

        # for GPU detection
        elif mode == "GPU":
            print("GPU mode")
            self.mode = 1
            self.yolo = Load_Yolo_model()
            self.input_size = 608
            self.score_threshold = threshold
            self.iou_threshold = 0.4

        self.det_time = 0
        self.pre_time = 0
        self.pro_time = 0
        
        self.labels = open(labels).read().strip().split("\n")
        self.logger = loguru.logger
        start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        log_file_name = '{:s}_{:s}.log'.format("LOG_FILE", start_time)
        log_file_path = os.path.join(os.getcwd(), log_file_name)

        self.logger.add("log.file",
                level='DEBUG',
                format="{time}{level}{message}"
                )
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
    def log_text(self, text):
        self.logger.log(text)

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

        
        detected_obj = DetectedPipeline(image)
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

    def make_prediction(self, image):

        height, width = image.shape[:2]
        detected_objects = []

        if self.mode == 0: #CPU detection
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
            
            if len(idxs) > 0:
                for i in idxs.flatten():

                    label = self.labels[classIDs[i]]
                    color = [int(c) for c in self.colors[classIDs[i]]]
                    detected_object = DetectedObject(
                        classIDs[i], confidences[i], boxes[i], label, color
                    )
                    detected_objects.append(detected_object)
        else: #GPU detection
            cv2_im = image
            original_frame = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

            preprocessing_start = time.time()
            image_data = image_preprocess(np.copy(original_frame), [self.input_size, self.input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            preprocessing_end = time.time()

            batched_input = tf.constant(image_data)
            result = self.yolo(batched_input)

            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0) 

            post_process_start = time.time()
            bboxes = postprocess_boxes(pred_bbox, original_frame, self.input_size, self.score_threshold)
            bboxes = nms(bboxes, self.iou_threshold, method='nms') 

            post_process_end = time.time()

            print("Pre processing time: ", preprocessing_end - preprocessing_start)
            print("Detection time: ", post_process_start - preprocessing_end)
            print("Post process time: ", post_process_end - post_process_start)


            for i, bbox in enumerate(bboxes):
                class_id = int(bbox[5])
                label = self.labels[class_id]
                color = [int(c) for c in self.colors[class_id]]
                confidence = bbox[4].item()
                boxes = bbox[:4].tolist()

                boxes[2] = int(boxes[2] - boxes[0])
                boxes[3] = int(boxes[3] - boxes[1])
                boxes[0] = int(boxes[0])
                boxes[1] = int(boxes[1])
                
                detected_object = DetectedObject(
                        class_id, confidence, boxes, label, color
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

def test_gpu():
    imageDet = ImageDetector(mode="GPU", weights="yolo-files/yolov4-tiny.weights", config_file="yolo-files/yolov4-tiny.cfg")

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
    print("Total detected time: ", end_time-start_time)

# test_gpu()
# test()
