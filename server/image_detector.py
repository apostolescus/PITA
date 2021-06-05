import uuid
import time
import typing

import numpy as np
import cv2
import csv

from shapely.geometry import Polygon
from darknet import (
    load_network,
    network_width,
    network_height,
    make_image,
    detect_image,
    copy_image_from_bytes,
    free_image,
)

from storage import DetectedPipeline, config_file
from storage import logger, get_poly_lines


class DetectedObject:
    """
    Class that contains information about a detected object.
    Stores information about: confidence score, bounding boxes,
    color to display bboxes, label and unique id.
    """

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
    and focal length: 
    https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

    Allows distance measurements using width or height.
    """

    def __init__(
        self,
        weights="yolo-files/yolov4.weights",
        config_file="yolo-files",
        mode="CPU",
        labels="yolo-files/coco.names",
        data_file="yolo-files/coco.data",
        average_size="yolo-files/average_size.csv",
        confidence=0.8,
        threshold=0.3,
    ):
        # load image sizes
        self._image_width = 640  # config_file["DETECTION"].getint("image_width")
        self._image_height = 480  # config_file["DETECTION"].getint("image_height")

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
            self.network, self.class_names, self.class_colors = load_network(
                config_file, data_file, weights
            )

            self.network_width = network_width(self.network)
            self.network_height = network_height(self.network)
            self._width_ratio = self._image_width / self.network_width
            self._height_ratio = self._image_height / self.network_height

        self.det_time = 0
        self.pre_time = 0
        self.pro_time = 0

        self.labels = open(labels).read().strip().split("\n")

        # get polygone to check for frontal vehicles
        self._FIX_POLY_LINES = Polygon(get_poly_lines("poly"))

        id_dictionary = {}

        # load dictionary containing average objects sizes
        with open(average_size, "r") as data:
            for line in csv.reader(data):
                element = line.pop(0)
                # remove -1 after correcting numbers in csv files (should start from 0, not from 1)
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

    def _get_distances(
        self, detected_list
    ) -> typing.Tuple[typing.Dict[int, float], typing.Dict[float, int]]:
        """
        Calculates distances and checks if vehicles in front.
        Returns two dictionaries. One for distances having key:distance, and value:object_id
        and another one for objects that were detected in front having key:object_id and value:0
        """

        distance_vector = {}
        frontal_objects = {}

        if detected_list:
            for detected_object in detected_list:

                # get object sizes
                x, y = detected_object.bbx[0], detected_object.bbx[1]
                w, h = detected_object.bbx[2], detected_object.bbx[3]

                # calculate object area
                object_area = w * h

                # calculate 75% from object's area
                object_area_min_val = 75 / 100 * object_area

                y_top = self._image_height - y - h

                # generate object points from x,y,w,h
                detected_object_points = [
                    (x, y_top),
                    (x + w, y_top),
                    (x + w, y_top + h),
                    (x, y_top + h),
                ]

                # generate polygone from object
                detected_object_polygone = Polygon(detected_object_points)

                # check if the two polygons intersect
                intersection = detected_object_polygone.intersects(self._FIX_POLY_LINES)

                if intersection:
                    # calculate intersection area
                    intersection_area = detected_object_polygone.intersection(
                        self._FIX_POLY_LINES
                    ).area

                    # if the intersection area is bigger then 75% of object area
                    # then the object is considered in the front of the driver

                    if intersection_area >= object_area_min_val:

                        # calculate the distance till the object
                        distance = self._get_distance(detected_object.id, w, h)

                        # add object distance to distance vector
                        distance_vector[distance] = detected_object.id
                        frontal_objects[detected_object.id] = 0

        return distance_vector, frontal_objects

    def detect(self, image) -> DetectedPipeline:
        """
        Public method used as entry point for image detection.

        Returns a DetectedPipeline object.
        """

        detected_obj = DetectedPipeline(image)
        detected_list = self._make_prediction(image)

        # if any objects were detected
        if detected_list:
            distances, frontal_list = self._get_distances(detected_list)
            detected_obj.frontal_distances = distances
            detected_obj.frontal_objects = frontal_list
            detected_obj.detected_objects = detected_list
            detected_obj.detected = True

        self.start = time.time()
        return detected_obj

    def _extract_boxes_confidences_classids(
        self, outputs
    ) -> typing.Tuple[
        typing.List[typing.Tuple[int, int, int, int]],
        typing.List[float],
        typing.List[int],
    ]:
        """
        Private method used fot CPU detection to filter and convert
        detected objects bounding boxes.
        """

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
                    box = detection[0:4] * np.array(
                        [
                            self._image_width,
                            self._image_height,
                            self._image_width,
                            self._image_height,
                        ]
                    )
                    centerX, centerY, w, h = box.astype("int")

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return boxes, confidences, classIDs

    def _make_prediction(self, image) -> typing.List[DetectedObject]:
        """
        Private method which passes the image through the CNN.
        Extracts class confidences and objects boxes.
        Applies NMS Suppression.

        Returns list of DetectedObjects.
        """

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
                outputs
            )

            # Apply Non-Max Suppression
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    label = self.labels[classIDs[i]]
                    color = self.class_colors[label]

                    detected_object = DetectedObject(
                        classIDs[i], confidences[i], boxes[i], label, color
                    )
                    detected_objects.append(detected_object)

        # GPU detection
        else:
            detections = self._detect_gpu(image)

            for label, conf, bboxes in detections:
                boxes = []

                class_id = int(self.labels.index(label))
                confidence = round(float(conf), 2)

                # filter confidence over certain step
                if confidence >= self.confidence:

                    # convert boxes values to integer to display
                    box = bboxes * np.array(
                        [
                            self._width_ratio,
                            self._height_ratio,
                            self._width_ratio,
                            self._height_ratio,
                        ]
                    )
                    centerX, centerY, w, h = box.astype("int")

                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes = [x, y, int(w), int(h)]
                    # extract color
                    color = self.class_colors[label]

                    # generate DetectedObject
                    detected_object = DetectedObject(
                        class_id, confidence, boxes, label, color
                    )
                    detected_objects.append(detected_object)

        return detected_objects

    def _detect_gpu(self, image):
        """
        Private method used to detect objects in
        image using GPU. Calling darknet.py methods.
        """

        darknet_image = make_image(self.network_width, self.network_height, 3)

        # convert color palette of image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image to fit the network
        img_resized = cv2.resize(
            img_rgb,
            (self.network_width, self.network_height),
            interpolation=cv2.INTER_LINEAR,
        )

        copy_image_from_bytes(darknet_image, img_resized.tobytes())

        # run model on darknet style image to get detections
        detections = detect_image(self.network, self.class_names, darknet_image)
        free_image(darknet_image)

        return detections

    def _get_distance(self, item_id, width, height, focal_length=596) -> float:
        """
        Calculates distance to the object based on the dictionary configuration.

        Returns the distance in cm as float.
        """

        # get object mode of distance calculation for file
        mode = int(self.average_size_dictionary[item_id][0])

        # get average object size
        value = self.average_size_dictionary[item_id][1]

        if mode == 0:
            denominator = height
        elif mode == 1:
            denominator = width
        elif mode == 2:
            max_val = max(width, height)
            denominator = max_val
        elif mode == 3:
            denominator = min(width, height)

        # calculate distance
        d = focal_length * float(value) / int(denominator)

        return d / 100  # convert from m to cm


def log_text(text):
    """ Log text using loguru. """
    logger.log("IMAGE_DETECTOR", text)


def test_gpu():
    """Function for testing GPU detection."""
    image_det = ImageDetector(
        mode="GPU",
        weights="yolo-files/yolov4-tiny.weights",
        config_file="yolo-files/yolov4-tiny.cfg",
    )

    image = cv2.imread("images/kite.jpg")
    print("Image shape: ", image.shape)
    start_time = time.time()
    detected = image_det.detect(image)
    end_time = time.time()

    for i in detected.detected_objects:
        print("--- new object ---")
        print("Class: ", i.id)
        print("Score: ", i.score)
        print("Label: ", i.label)

    print("Detected objects on GPU: ", detected)
    print("Total detected time: ", end_time - start_time)
