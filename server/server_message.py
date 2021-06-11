"""
Main component of the PITA server side application.
Manages the communication with the client and implements
pipeline and parallel processing of the client requests.
"""
import selectors
import struct
import io
import json
import base64
import uuid
import threading
import time
from queue import Queue
import cv2
import numpy as np

from image_detector import ImageDetector
from video_manager_wrapper import VideoManagerWrapper
from alerter import Alerter, Update
from lane_detector import LaneDetector
from storage import StoppableThread
from storage import logger, config_file, timer, debug_mode
from storage import set_poly_lines

lane_detection = False

# build pipeline
image_detection_queue = Queue(1)
lane_detection_queue = Queue(1)
alerter_queue_image = Queue(1)
alerter_queue_lane = Queue(1)
alerter_queue = Queue(1)
results_queue = Queue(1)

# debugging
average_time = 0
average_time_counter = 0


class AlertThread(StoppableThread):
    """
    Wrapper for Alert Class to Stoppable Thread.
    """

    def __init__(self, name, client_uid):
        super(AlertThread, self).__init__(name)
        self.alerter = Alerter(client_uid)

    def update(self, update):
        self.alerter.update(update)

    def run(self):
        global lane_detection_queue

        while not self.stopevent.isSet():

            # extract detection result from image queue
            image_id, detection_results, image = alerter_queue_image.get()

            # extract lane detection from lane queue
            if lane_detection:
                lane_id, lanes = alerter_queue_lane.get()
            else:
                lane_id = image_id
                lanes = []

            # extract gps infos from alert queue
            alert_id, gps_infos = alerter_queue.get()

            if image_id == lane_id == alert_id:

                # send frame to video manager
                self.alerter.video_manager.record(image)

                # check safety based on extracted informations
                self.alerter.check_safety(detection_results, gps_infos)

            results_queue.put([detection_results, lanes])

        threading.Thread.join(self)


class LaneDetectorThread(StoppableThread):
    """
    Stoppable Thread wrapper for Lane Detector class.
    """

    logger.log("LANE_DETECTOR", "START")

    def run(self):

        lane_detector = LaneDetector()

        while not self.stopevent.isSet():
            try:
                # fetch image from pipeline
                detect_id, image = lane_detection_queue.get()

                if timer:
                    start_time = time.time()

                # perform lane detection
                detected_lines = lane_detector.detect_lane(image)

                # craft response and send in pipeline
                new_response = [detect_id, detected_lines]

                if timer:
                    end_time = time.time()
                    logger.log(
                        "LANE_DETECTOR",
                        "Total lane det time: " + str(end_time - start_time),
                    )

                alerter_queue_lane.put(new_response)
            except:
                continue

        logger.log("LANE_DETECTOR", "STOP")
        threading.Thread.join(self)


class ImageObjectDetectorThread(StoppableThread):
    """
    Stoppable Thread wrapper for Image Detector class.
    """

    def __init__(self, name):
        super(ImageObjectDetectorThread, self).__init__(name)

        self.yolo_weights = config_file["DETECTION"]["yolo_weights"]
        self.yolo_cfg = config_file["DETECTION"]["yolo_cfg"]
        self.detection_mode = config_file["DETECTION"]["mode"]
        self.labels = config_file["DETECTION"]["labels"]
        self.avg_size_csv = config_file["DETECTION"]["avg_file"]
        self.confidence = config_file["DETECTION"].getfloat("confidence")
        self.threshold = config_file["DETECTION"].getfloat("threshold")
        self.data_file = config_file["DETECTION"]["data_file"]
        self.last_results = []
        self.last_detection = 0.1

    def run(self):

        imageDetector = ImageDetector(
            weights=self.yolo_weights,
            config_file=self.yolo_cfg,
            mode=self.detection_mode,
            labels=self.labels,
            data_file=self.data_file,
            average_size=self.avg_size_csv,
            confidence=self.confidence,
            threshold=self.threshold,
        )

        while not self.stopevent.isSet():
            # fetch image from pipeline
            detect_id, image = image_detection_queue.get()

            if timer:
                start = time.time()

            if time.time() - self.last_detection > 0.1:
                # perform object detection
                detection_results = imageDetector.detect(image)
                self.last_result = [detect_id, detection_results, image]
                self.last_detection = time.time()

                if timer:
                    end = time.time()
                    logger.log(
                        "IMAGE_DETECTOR", "detection time is: " + str(end - start)
                    )

            alerter_queue_image.put(self.last_result)

        threading.Thread.join(self)


class Message:
    """
    Server Side Message Class.
    Contains messages templates to communicate with the client.
    """

    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._read_header = True
        self._request_done = False
        self._package_received = False
        self._data = b""
        self._results = None
        self._waiting_images = Queue(3)
        self._detection_results = Queue(1)

        # get first message
        # and poligone
        self._first_detection = True

    def process_message(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def _set_selector_events_mask(self, mode):
        """
        Set selector to listen for events.
        Mode is 'r', 'w', or 'rw'.
        """

        if mode == "r":
            if debug_mode:
                logger.log("SERVER", "Server switched to read mode")
            events = selectors.EVENT_READ
        elif mode == "w":
            if debug_mode:
                logger.log("SERVER", "Server switched to write mode")
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            if debug_mode:
                logger.log("SERVER", "Server switched to read/write mode")
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            logger.error(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _json_decode(self, json_bytes, encoding="utf-8"):
        """
        Private method to decode json ansewer.
        """

        tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return obj

    def _json_encode(self, stream):
        """
        Private method to encode json.
        """

        return json.dumps(stream, ensure_ascii=False).encode()

    def _read(self) -> None:
        """
        Private method to read bytestream form socket.

        Stream is recieved in 2048*8 chunks.
        If the connection is interruped call video manager
        to save the recorded video.
        """

        try:
            data = self.sock.recv(2048 * 8)
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                # call stop method video manager
                video_manager = VideoManagerWrapper.getInstance()
                video_manager.stop()
                raise RuntimeError("Peer closed.")

    def _get_header(self):
        """
        Private method that extracts header from
        the bytestream.
        Header length is in the first 2 bytes.
        """

        header_len = 2

        if len(self._recv_buffer) >= header_len:
            self._header_len = struct.unpack("<H", self._recv_buffer[:header_len])[0]
            self._recv_buffer = self._recv_buffer[header_len:]

    def _process_header(self):
        """
        Private method that reads and decodes
        header content.
        """

        header_len = self._header_len

        if len(self._recv_buffer) >= header_len:

            self.json_header = self._json_decode(self._recv_buffer[:header_len])
            self._recv_buffer = self._recv_buffer[header_len:]
            self._read_header = False

    def _process_request(self):
        """
        Process recieved bytes and split them based on
        the message type.
        Process first message content, load detection lines
        and start processing threads.
        """
        request_type = self.json_header["request-type"]

        # if it's the first message
        if self._first_detection:

            # load lines for line/frontal detection
            np_lines = self.json_header["np-lines"]
            poly_lines = self.json_header["poly-lines"]
            client_uid = self.json_header["user-uid"]

            set_poly_lines(poly_lines, np_lines)

            # start pipeline threads
            ImageObjectDetectorThread("image_detector").start()
            AlertThread("alerter", client_uid).start()

            self._first_detection = False

        if request_type == "DETECT":
            content_len = self.json_header["content-len"]

            if len(self._recv_buffer) >= content_len:

                self._data = self._recv_buffer[:content_len]
                self._recv_buffer = self._recv_buffer[content_len:]
                self._read_header = True

        elif request_type == "UPDATE":

            self._data = self._recv_buffer[: self._header_len]
            self._recv_buffer = self._recv_buffer[self._header_len :]
            self._read_header = True

    def _process_results(self):
        """
        Generate a dictionary containing detected
        objects infos, alerts and lines.
        Set generated dictionary as self._result variable.
        """

        detected_obj = self._results[0]
        dictionary = {}

        if debug_mode:
            logger.log("SERVER", "Processing Results")

        if lane_detection:
            lines = self._results[1]
            if len(lines) >= 1:
                dictionary["lines"] = lines.tolist()
            else:
                dictionary["lines"] = None
        else:
            dictionary["lines"] = None

        if detected_obj:
            if detected_obj.detected:

                formatted_list = []
                detected_list = detected_obj.detected_objects

                frontal_object = detected_obj.frontal_objects
                distance_vector = {
                    v: k for k, v in detected_obj.frontal_distances.items()
                }

                alert = detected_obj.alert

                for obj in detected_list:

                    obj_dictionary = {}
                    obj_dictionary["coordinates"] = obj.bbx

                    color = obj.color
                    label = obj.label
                    score = obj.score

                    obj_id = obj.id

                    if obj_id in frontal_object:
                        distance = distance_vector[obj_id]
                        obj_dictionary["distance"] = round(distance, 2)

                    obj_dictionary["color"] = color
                    obj_dictionary["label"] = label
                    obj_dictionary["score"] = score

                    formatted_list.append(obj_dictionary)

                dictionary["detected_objects"] = formatted_list
                dictionary["danger"] = detected_obj.danger

                if alert:
                    dictionary["alert"] = alert
                else:
                    dictionary["alert"] = ""

                if detected_obj.safe_distance != 0:
                    dictionary["safe_distance"] = round(detected_obj.safe_distance, 2)
            else:
                dictionary["detected_objects"] = None
                dictionary["danger"] = None
        else:
            dictionary["detected_objects"] = None
            dictionary["danger"] = None

        self._results = dictionary

    def read(self):
        """
        Main method that calls the underlaying methods
        to fetch image and gps coordinates from client and
        send them to the pipeline.

        Extracts information differently based on the message type.
        """
        global lane_detection

        self._read()

        # check if is image or update request
        if self._read_header:
            self._get_header()
            self._process_header()

        self._process_request()

        if self._read_header:
            # the full message was received
            # process the content

            # extract the uuid from request
            self._uuid = self.json_header["uuid"]

            if self.json_header["request-type"] == "DETECT":

                if timer:
                    global average_time, average_time_counter
                    send_time = self.json_header["time"]
                    dif = time.time() - send_time

                    if average_time_counter > 100:
                        logger.log(
                            "SERVER",
                            "Average time is: "
                            + str(average_time / average_time_counter),
                        )
                    else:
                        average_time += dif
                        average_time_counter += 1

                    logger.log("SERVER", "From client to server: " + str(dif))

                # print(str(time.time()- self.json_header["time"]))

                # extract speed and gps coordinates
                gps_speed = self.json_header["speed"]
                gps_lat = self.json_header["lat"]
                gps_lon = self.json_header["lon"]

                gps_infos = [gps_speed, gps_lat, gps_lon]

                # extract image from byte stream
                jpg_original = base64.b64decode(self._data)
                jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                img = cv2.imdecode(jpg_as_np, flags=1)

                detect_id = uuid.uuid4()
                # add image to processing queue
                image_detection_queue.put((detect_id, img))

                # add image to line detection queue
                if lane_detection:
                    lane_detection_queue.put((detect_id, img))

                alerter_queue.put((detect_id, gps_infos))

                # clear reciving buffer
                self._data = b""

                self._results = results_queue.get()
                self._process_results()

                self._set_selector_events_mask("w")

            elif self.json_header["request-type"] == "UPDATE":

                if debug_mode:
                    logger.log("SERVER", "update request")

                update_mode = self.json_header["update"]

                if update_mode == 1:
                    # proceed to full update
                    car_type = self.json_header["car_type"]
                    weather = self.json_header["weather"]
                    experience = self.json_header["experience"]
                    reaction_time = self.json_header["reaction_time"]
                    record_mode = self.json_header["record_mode"]
                    update = Update(
                        record_mode,
                        car_type,
                        weather,
                        experience,
                        reaction_time,
                    )
                    for thread in threading.enumerate():
                        if thread.name == "alerter":
                            thread.update(update)

                else:
                    # only lane updater was switched
                    lane_det = self.json_header["lane"]

                    if debug_mode:
                        logger.log("SERVER", "Lane detector is: " + str(lane_det))
                        logger.log(
                            "SERVER",
                            "Global variable lane_detection is: " + str(lane_detection),
                        )
                    # print("Global variable lane_detection is: ", lane_detection)

                    if lane_det:
                        if lane_detection is False:
                            logger.log("SERVER", "Stargin lane detector")
                            LaneDetectorThread("lane_detector").start()
                            lane_detection = True
                    else:
                        if lane_detection:
                            for thread in threading.enumerate():
                                if thread.name == "lane_detector":
                                    logger.log("SERVER", "Stopping Lane Detector")
                                    thread.stop()

                            lane_detection = False

                self._results = ""
                self._set_selector_events_mask("w")

    def _generate_request(self):
        """
        Private method that generates the header and
        encapsulates the message in order to send it to the client.
        """

        response = self._results
        encoded_response = b""

        # objects succesfully detected
        # send response
        if len(response) >= 1:
            encoded_response = self._json_encode(response)
            header = {
                "response-type": "DETECTED",
                "content-len": len(encoded_response),
                "time": time.time(),
                "uuid": self._uuid,
            }
        # no object was detected
        else:
            header = {"time": time.time(), "response-type": "EMPTY", "uuid": self._uuid}

        encoded_header = self._json_encode(header)
        message_hdr = struct.pack("<H", len(encoded_header))

        msg = message_hdr + encoded_header + encoded_response

        self._send_buffer += msg
        self._request_done = True

    def _write(self):
        """
        Low level private method that
        sends the bytestream to the client.
        """

        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
                if debug_mode:
                    logger.log("SERVER", "Sended: " + str(sent))
                self._request_done = False
            except BlockingIOError:
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]
        else:
            if debug_mode:
                logger.log("SERVER", "No content in send buffer")

    def write(self):
        """
        High level public method that manages the writing process to the client.
        """

        if debug_mode:
            logger.log("SERVER", "Writing to client")

        if self._request_done is False:
            self._generate_request()

        self._write()

        if not self._send_buffer:
            self._set_selector_events_mask("r")

    def close(self):
        """
        Private method called when a client is unregistered from the socket list.
        """

        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            logger.error(
                "error: selector.unregister() exception for", f"{self.addr}: {repr(e)}"
            )

        try:
            self.sock.close()
        except OSError as e:
            logger.error(
                "error: socket.close() exception for", f"{self.addr}: {repr(e)}"
            )
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None
