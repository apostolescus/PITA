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
from lane_detector import LaneDetector
from storage import StoppableThread
from storage import DetectedPipeline

lane_detection = False

image_detection_queue = Queue(1)
lane_detection_queue = Queue(1)
alerter_queue = Queue(1)
results_queue = Queue(1)

# debugging
debug_mode = False
timing = False
average_time = 0
average_time_counter = 0

class AlertThread(StoppableThread):
    def __init__(self, name):

        super(AlertThread, self).__init__(name)
        height = 1080
        self.alerter = Alerter([(340, height - 150), (920, 550), (1570, height - 150)])

    def update(self, update):
        self.alerter.update(update)

    def run(self):
        global lane_detection_queue

        self.stop = False

        while not self.stopevent.isSet():
            
            third_step = alerter_queue.get()
            
            self.alerter.video_manager.record(third_step[0])
            self.alerter.check_safety(third_step)

            results_queue.put(third_step)

        threading.Thread.join(self)

    def update_data(self):
        self.alerter.update_alert_logger()

class LaneDetectorThread(StoppableThread):

    def run(self):

        lane_detector = LaneDetector()

        while not self.stopevent.isSet():
            try:
                image_det = lane_detection_queue.get()
                image = image_det[0]
                start_time = time.time()
                detected_lines = lane_detector.detect_lane(image)

                new_response = [image, image_det[1], detected_lines, image_det[2]]
                end_time = time.time()
                print("Total lane det time: ", end_time-start_time)
                alerter_queue.put(new_response)
            except:
                continue

        # try:
        #     value = lane_detection_queue.get_nowait()   
        # except Empty:
        #     print("No object in queue")
        # else:
        #     print("There was an object in queue")
        #     alerter_queue.put(value)

        #
        # print("LANE DETECTION STOPPED")
        # threading.Thread.join(self)


class ImageObjectDetectorThread(StoppableThread):
    def run(self):

        imageDetector = ImageDetector(
            "yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg"
        )

        # imageDetector = ImageDetector("yolo-files/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", "yolo-files/yolov4-tiny.cfg", model = "tflite")

        while not self.stopevent.isSet():

            (read_image, gps_infos) = image_detection_queue.get()

            start = time.time()
            detection_results = imageDetector.detect(read_image)
            result = [read_image, detection_results, gps_infos]
            end = time.time()
            print("Total detection time: ", end-start)

            #print("Image detection time: ", end-start)
            #print("Imae detection lane_value: ", lane_detection)
            if lane_detection:
                #print("-- image detector --- putting in Lane Detection Queue")
                lane_detection_queue.put(result)
            else:
                #print("-- image detector --- putting in Alerter Queue")
                alerter_queue.put(result)

        threading.Thread.join(self)

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
            data = self.sock.recv(2048*8)
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

    def _process_results(self):

        detected_obj = self._results[1]
        dictionary = {}

        if debug_mode:
            print("Processing results")

        if lane_detection:
            lines = self._results[2]
            if len(lines)>=1:
                dictionary["lines"] = lines.tolist()
            else:
                dictionary["lines"] = None
        else:
            dictionary["lines"] = None

        if detected_obj:
            if detected_obj.detected:

                formatted_list = []
                detected_list = detected_obj.detected_objects
                alerts = detected_obj.alerts

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
                
                if alerts:
                    dictionary["alerts"] = alerts
                else:
                    dictionary["alerts"] = None
            else:
                dictionary["detected_objects"] = None
                dictionary["danger"] = None
        else:
            dictionary["detected_objects"] = None
            dictionary["danger"] = None

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

            if self.json_header["request-type"] == "DETECT":
                # self._uuid = self.json_header["uuid"]
                if timing:
                    global average_time, average_time_counter
                    send_time = self.json_header["time"]
                    dif = time.time() - send_time

                    if average_time_counter > 100:
                        print("Average time is: ", average_time/ average_time_counter)
                    else:
                        average_time += dif
                        average_time_counter += 1

                    print("From client to server: " + str(dif))
                #print(str(time.time()- self.json_header["time"]))

                # extract speed and gps coordinates
                gps_speed = self.json_header["speed"]
                gps_lat = self.json_header["lat"]
                gps_lon = self.json_header["lon"]
                
                gps_infos = [gps_speed, gps_lat, gps_lon]

                # extract image from byte stream
                jpg_original = base64.b64decode(self._data)
                jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
                img = cv2.imdecode(jpg_as_np, flags=1)

                # add image to processing queue
                image_detection_queue.put((img, gps_infos))

                # clear reciving buffer
                self._data = b""

                self._results = results_queue.get()
                
                if debug_mode:
                    print("Object poped from queue: ", self._results)

                
                self._process_results()
                
                self._set_selector_events_mask("w")

            elif self.json_header["request-type"] == "UPDATE":
                if debug_mode:
                    print("Update request")
               
                update_mode = self.json_header["update"]

                if update_mode == 1:
                    # proceed to full update
                    car_type = self.json_header["car_type"]
                    weather = self.json_header["weather"]
                    experience = self.json_header["experience"]
                    reaction_time = self.json_header["reaction_time"]
                    record_mode = self.json_header["record_mode"]
                    update = Update(
                        1,
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
                    global lane_detection

                    lane_det = self.json_header["lane"]
                    #print("Land detector is: ", lane_det)
                    if debug_mode:
                        print("Lane detector is: ", lane_det)
                        print("Lane detection on sever is: ", lane_detection)

                    if lane_det:
                        if lane_detection is False:
                            #print("Before thread starting: ", threading.active_count())
                            lane_thread = LaneDetectorThread("lane_detector").start()
                            lane_detection = True
                            #print("After thread starting: ", threading.active_count())
                    else:
                        if lane_detection is True:
                            #print("Before stopping thread: ", threading.active_count())
                            for thread in threading.enumerate():
                                if thread.name == "lane_detector":
                                    print("Stopping thread")
                                    thread.join()
                            lane_detection = False
                            #rint("After stop threads running: ", threading.active_count())
                    #print("Lane detection variable: ", lane_detection)

                # update local detector and alerter
               
                self._results = ""
                self._set_selector_events_mask("w")

    def _generate_request(self):

        response = self._results
        print("Response: ", response)
        encoded_response = b""
        # objects succesfully detected
        # send response

        if len(response) >= 1:
            encoded_response = self._json_encode(response)
            header = {
                "response-type": "DETECTED",
                "content-len":len(encoded_response),
                "time":time.time()
            }
           
        else:
            header = {
                "time":time.time(),
                "response-type": "EMPTY",
                }

        encoded_header = self._json_encode(header)
        message_hdr = struct.pack("<H", len(encoded_header))

        msg = message_hdr + encoded_header + encoded_response

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
            if debug_mode:
                print("--- main thread --- No content in send buffer")

    def write(self):
        if debug_mode:
            print("Writing to client")
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
