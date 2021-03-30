import selectors
import struct
import io
import json
import base64
import cv2
import copy
import numpy as np

import uuid
import threading
import time
from datetime import datetime
from queue import Queue, Empty

from image_detector import ImageDetector, DetectedObject
from alerter import Alerter, Update
from storage import StoppableThread
from storage import DetectedPipeline

lane_detection = False

# waiting_image_queue = Queue(1)
image_detection_queue = Queue(1)
lane_detection_queue = Queue(1)
alerter_queue = Queue(1)
results_queue = Queue(1)

# debugging
debug_mode = False


class AlertThread(StoppableThread):
    def __init__(self, name):

        super(AlertThread, self).__init__(name)
        height = 1080
        self.alerter = Alerter([(340, height - 150), (920, 550), (1570, height - 150)])

    def update(self, update):
        # rint("UPDATINNNNNNG")
        # print("update object: ", update)
        self.alerter.update(update)

    def run(self):
        global lane_detection_queue

        # starts GPS thread
        # gps = GPS("GPSThread")
        # gps.start()

        self.stop = False

        while not self.stopevent.isSet():

            # if update is True:
            #     videoManager.stop()
            #     self.alerter.update()
            #     videoManager.update()
            #     update = False

            third_step = alerter_queue.get()

            # self.alerter.video_manager.record(image)
            self.alerter.check_safety(third_step)

            # drawn_image = self.alerter.draw_image(res[0], res[1], lines)
            results_queue.put(third_step)

        print("Alerter stopped")

    def update_data(self):
        self.alerter.update_alert_logger()


class ImageObjectDetectorThread(StoppableThread):
    def run(self):

        imageDetector = ImageDetector(
            "yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg"
        )

        # imageDetector = ImageDetector("yolo-files/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", "yolo-files/yolov4-tiny.cfg", model = "tflite")

        while True:

            read_image = image_detection_queue.get()

            detection_results = imageDetector.detect(read_image)
            result = [read_image, detection_results]

            if lane_detection:
                lane_detection_queue.put(result)
            else:
                alerter_queue.put(result)


image_thread = ImageObjectDetectorThread("image_detector").start()
alerter_thread = AlertThread("alerter").start()


class Message:
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

    def process_message(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            if debug_mode:
                print("--- main thread --- Server switched to read mode")
            events = selectors.EVENT_READ
        elif mode == "w":
            if debug_mode:
                print("--- main thread --- Server switched to write mode")
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            if debug_mode:
                print("--- main thread --- Server switched to read/write mode")
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _json_decode(self, json_bytes, encoding="utf-8"):

        tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return obj

    def _json_encode(self, stream):
        return json.dumps(stream, ensure_ascii=False).encode()

    def _read(self):
        try:
            data = self.sock.recv(2048)
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def _get_header(self):
        header_len = 2

        if len(self._recv_buffer) >= header_len:
            self._header_len = struct.unpack("<H", self._recv_buffer[:header_len])[0]
            self._recv_buffer = self._recv_buffer[header_len:]

    def _process_header(self):
        header_len = self._header_len

        if len(self._recv_buffer) >= header_len:

            self.json_header = self._json_decode(self._recv_buffer[:header_len])
            self._recv_buffer = self._recv_buffer[header_len:]
            self._read_header = False

    def _process_request(self):

        request_type = self.json_header["request-type"]

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

    def _signal_working(self):

        self._set_selector_events_mask("w")
        header = {
            "response-type": "SEND",
        }

        encoded_header = self._json_encode(header)
        message_hdr = struct.pack("<H", len(encoded_header))

        msg = message_hdr + encoded_header

        self._send_buffer += msg
        self._request_done = True

        self._write()
        self._read_header = True
        self._set_selector_events_mask("r")

    def _process_results(self):

        detected_obj = self._results[1]
        dictionary = {}

        if detected_obj:
            if detected_obj.detected:

                formatted_list = []
                detected_list = detected_obj.detected_objects

                for obj in detected_list:

                    obj_dictionary = {}
                    obj_dictionary["coordinates"] = obj.bbx

                    color = obj.color
                    label = obj.label
                    score = obj.score

                    obj_dictionary["color"] = color
                    obj_dictionary["label"] = label
                    obj_dictionary["score"] = score

                    formatted_list.append(obj_dictionary)

                dictionary["detected_objects"] = formatted_list
                dictionary["danger"] = detected_obj.danger

        self._results = dictionary

    def read(self):

        self._read()

        # check if is image or update request
        if self._read_header:
            self._get_header()
            self._process_header()

        self._process_request()

        if self._read_header:
            # the full message was received
            # process it's content
            # print("--- main thread --- Full message was received")

            if self.json_header["request-type"] == "DETECT":

                # extract image from byte stream
                jpg_original = base64.b64decode(self._data)
                jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                img = cv2.imdecode(jpg_as_np, flags=1)

                # add image to processing queue
                image_detection_queue.put(img)

                # clear reciving buffer
                self._data = b""

                self._results = results_queue.get()

                if self._results:
                    self._process_results()

                self._set_selector_events_mask("w")

            elif self.json_header["request-type"] == "UPDATE":

                update_mode = self.json_header["update"]

                if update_mode == 1:
                    car_type = self.json_header["car_type"]
                    weather = self.json_header["weather"]
                    experience = self.json_header["experience"]
                    reaction_time = self.json_header["reaction_time"]
                    record_mode = self.json_header["record_mode"]
                    lane_det = self.json_header["lane"]
                    update = Update(
                        1,
                        lane_det,
                        record_mode,
                        car_type,
                        weather,
                        experience,
                        reaction_time,
                    )

                else:
                    lane_det = self.json_header["lane"]
                    update = Update(0, lane_det)

                # update local detector and alerter
                for thread in threading.enumerate():
                    if thread.name == "alerter":
                        thread.update(update)

                self._results = ""
                self._set_selector_events_mask("w")

    def _generate_request(self):

        response = self._results

        # objects succesfully detected
        # send response

        if len(response) >= 1:
            encoded_response = self._json_encode(response)
            header = {
                "danger": 0,
                "response-type": "DETECTED",
                # used for debugging
                "response-id": str(uuid.uuid4()),
                "content-len": len(encoded_response),
            }

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))

            msg = message_hdr + encoded_header + encoded_response
        else:
            header = {"response-type": "EMPTY", "response-id": str(uuid.uuid4())}

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))

            msg = message_hdr + encoded_header

        self._send_buffer += msg
        self._request_done = True

    def _write(self):

        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
                if debug_mode:
                    print("--- main thread --- Sended :", sent)
                self._request_done = False
            except BlockingIOError:
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]
        else:
            print("--- main thread --- No content in send buffer")

    def write(self):

        if self._request_done is False:
            self._generate_request()

        self._write()

        if not self._send_buffer:
            self._set_selector_events_mask("r")

    def close(self):

        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            print(
                "error: selector.unregister() exception for",
                f"{self.addr}: {repr(e)}",
            )

        try:
            self.sock.close()
        except OSError as e:
            print(
                "error: socket.close() exception for",
                f"{self.addr}: {repr(e)}",
            )
        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None
