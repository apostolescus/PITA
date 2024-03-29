"""
Module responsible for handling communication messages between server and client.
"""
import time
import io
import base64
import sys
import uuid
from queue import Empty, Full

import selectors
import json
import struct
import numpy as np
import cv2

# import locals
from screen_manager import captured_image_queue, result_queue
from storage import toggle_update_message, get_update_message, config_file
from storage import UISelected, get_switch_sound, gps_queue, logger
from storage import last_alert_queue, distance_queue
from storage import load_polygone_lines, safe_distance_queue
from sound_manager import SoundManager

# globals
lane_detection = False

# debugg
counter = config_file["DEBUG"].getint("video_frames")
DEBUG = bool(config_file["DEBUG"].getboolean("verbose"))
TIME = bool(config_file["DEBUG"].getboolean("time"))

# measure total travel time
uuid_dict = {}


class Message:
    """Implements own application protocol with server."""

    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._request_done = False
        self._read_header = True
        self._send_buffer = b""
        self._recv_buffer = b""
        self._current_image = None

        # used for time measurements
        self._average_time = 0
        self._average_time_counter = 0

        # delay time to send message
        self._last_message = 0
        self._last_response = 0

        # save last gps infos
        self._last_lat: float = 0
        self._last_lon: float = 0
        self._last_speed: int = 0

        # only for the first time send polygone
        self._start = True
        self._polygone_lines = []
        self._np_lines = []
        self._DISPLAY_TRIANGLE: bool = config_file["FRAME"].getboolean("display")

        # manager of alert sounds
        self._sound_manager = SoundManager()

        # alert display only list
        self._alert_list = config_file["ALERT"]["show_alerts"].split(",")
        
        # only for lane detection comparision
        if counter != 0:
            self.out = cv2.VideoWriter(
                "out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 6, (640, 480)
            )
            self.saved = False
        else:
            self.saved = True

    def process_message(self, mask):
        if mask & selectors.EVENT_WRITE:
            self.write()
        if mask & selectors.EVENT_READ:
            self.read()

    def _json_decode(self, json_bytes, encoding="utf-8"):
        """decodes bytes to json"""
        tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return obj

    def _json_encode(self, stream):
        """encodes json to bytestream"""
        return json.dumps(stream, ensure_ascii=False).encode()

    def _set_selector_events_mask(self, mode: str):
        """'Set selector to listen for events: mode is 'r', 'w', or 'rw'."""

        if mode == "r":
            if DEBUG:
                logger.debug("---main thread--- client switched to listen mode")
            events = selectors.EVENT_READ

        elif mode == "w":
            if DEBUG:
                logger.debug("---main thread--- client switched to write mode")
            events = selectors.EVENT_WRITE

        elif mode == "rw":
            if DEBUG:
                logger.debug("---main thread--- client switched to read/write mode")
            events = selectors.EVENT_READ | selectors.EVENT_WRITE

        else:
            logger.exception(f"Invalid events mask mode {repr(mode)}.")

        self.selector.modify(self.sock, events, data=self)

    def _generate_request(self):
        """Method that generates a request for the server."""
        msg: str = ""

        # generate and add unique id to dictionary
        unique_id = uuid.uuid4()
        uuid_dict[str(unique_id)] = time.time()

        if DEBUG:
            print("--- generate_request --- entered in generate request")

        # checks if the client made any updates in GUI
        if get_update_message():
            global lane_detection

            toggle_update_message()

            if DEBUG:
                print("--- generate_request --- update message")

            # update settings
            if UISelected.updated:
                header = {
                    "request-type": "UPDATE",
                    "update": 1,
                    "car_type": UISelected.car_type,
                    "weather": UISelected.weather,
                    "experience": UISelected.experience,
                    "record_mode": UISelected.rec_mode,
                    "reaction_time": UISelected.reaction_time,
                    "uuid": str(unique_id),
                }
                UISelected.updated = False
            else:
                header = {
                    "request-type": "UPDATE",
                    "update": 0,
                    "time": time.time(),
                    "lane": UISelected.lane_detection,
                    "uuid": str(unique_id),
                }

            # if first message send polygone lines
            if self._start:
                self._np_lines, poly_line = load_polygone_lines()
                header["np-lines"] = self._np_lines
                header["poly-lines"] = poly_line
                header["user-uid"] = config_file["DATABASE"]["uid"]
                print("Header: ", header)
                self._start = False

            # update lane detection
            lane_detection = UISelected.lane_detection

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header
        else:
            # extract iamge from queue
            self._current_image = captured_image_queue.get()

            # encode image in base64
            content = base64.b64encode(cv2.imencode(".jpg", self._current_image)[1])

            if DEBUG:
                logger.debug("--- generate_request --- content len: ", len(content))

            # get GPS infos
            try:
                gps_infos = gps_queue.get_nowait()
                speed = gps_infos[0]
                lat = gps_infos[1]
                lon = gps_infos[2]

                if speed != 0:
                    self._last_speed = speed
                
                if lon != 0:
                    self._last_lon = lon
                
                if lat != 0:
                    self._last_lat = lat

            except Empty:
                pass

            # create header dictionary
            header = {
                "request-type": "DETECT",
                "time": time.time(),
                "speed": self._last_speed,
                "lat": self._last_lat,
                "lon": self._last_lon,
                "uuid": str(unique_id),
                "content-len": len(content),
            }

            # if DEBUG:
            #     print("Request id: ", header["request-id"])

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header + content

        self._send_buffer += msg
        self._request_done = True

        return msg

    def write(self):

        if DEBUG:
            logger.debug("--- write --- entering write function")

        if self._request_done is False:

            if DEBUG:
                logger.debug("--- write --- request done is False")

            self._generate_request()

        # write all to buffer
        while True:
            if self._write():
                break

        if DEBUG:
            logger.debug("--- write --- processing request")

        if self._request_done:

            if DEBUG:
                logger.debug("--- write ---  request done")

            if not self._send_buffer:

                if DEBUG:
                    logger.debug("--- write ---  send buffer is empty")

                self._set_selector_events_mask("r")

    def _write(self) -> bool:

        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
            except BlockingIOError:
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]

        # if the buffer is not empty repeat sending

        return not self._send_buffer

    def _get_header(self):
        """ extract header from a request"""

        # header len is specified in the first 2 bytes
        header_len = 2

        if len(self._recv_buffer) >= header_len:
            self._header_len = struct.unpack("<H", self._recv_buffer[:header_len])[0]
            self._recv_buffer = self._recv_buffer[header_len:]

    def _process_header(self):
        """ decode header from bytestream and save in buffer"""
        header_len = self._header_len

        if len(self._recv_buffer) >= header_len:
            self.json_header = self._json_decode(self._recv_buffer[:header_len])
            self._recv_buffer = self._recv_buffer[header_len:]

    def _process_request(self):

        json_type = self.json_header["response-type"]
        uuid = self.json_header["uuid"]

        if TIME:

            start_time = uuid_dict[uuid]
            current_time = time.time()
            self._average_time = self._average_time + (current_time - start_time)
            self._average_time_counter += 1

            if self._average_time_counter % 100 == 0:
                logger.info(
                    "Average time in "
                    + str(self._average_time_counter)
                    + " messages: "
                    + str(current_time - start_time)
                )

        if DEBUG:
            logger.debug("--- main_thread --- processing request")

        if json_type == "EMPTY":

            if DEBUG:
                logger.debug("--- main_thread --- empty request")

            self._read_header = True
            self._request_done = False

            self._set_selector_events_mask("w")

        elif json_type == "DETECTED":

            if DEBUG:
                logger.debug("--- main_thread --- detected request")

            self._request_done = False
            content_len = self.json_header["content-len"]

            if len(self._recv_buffer) >= content_len:

                data = self._recv_buffer[:content_len]

                self._recv_buffer = self._recv_buffer[content_len:]
                self._read_header = True

                decoded_response = self._json_decode(data)

                self._display_image(decoded_response)

                self._set_selector_events_mask("w")

        result_queue.put(self._current_image)

    def _display_image(self, response):
        global counter

        obj_list = response["detected_objects"]
        danger = response["danger"]
        line = response["lines"]

        # displays and notify driver
        try:
            alert = response["alert"]
            if alert and alert in self._alert_list:
                try:
                    last_alert_queue.put(alert)
                    self._sound_manager.play_sound_custom_notification(alert)
                except Full:
                    pass

        except KeyError:
            pass

        # fetch safe distance
        try:
            safe_distance = response["safe_distance"]
            try:
                safe_distance_queue.put_nowait(safe_distance)
            except Full:
                pass
        except KeyError:
            pass

        switch_sound = get_switch_sound()

        if danger == 1 and switch_sound:
            self._sound_manager.play_sound()

        image = self._current_image

        if obj_list is not None:
            for obj in obj_list:

                x, y = obj["coordinates"][0], obj["coordinates"][1]
                w, h = obj["coordinates"][2], obj["coordinates"][3]

                color = obj["color"]
                label = obj["label"]
                score = obj["score"]

                text = "{}: {}".format(label, score)

                try:
                    distance = obj["distance"]
                    try:
                        distance_queue.put_nowait((label, distance))
                    except Full:
                        pass
                except KeyError:
                    pass

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

                self._current_image = image

        if line is not None:
            line_image = np.zeros_like(image)

            for x1, y1, x2, y2 in line:
                try:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 190, 255), 11)
                except:
                    return None
            self._current_image = cv2.addWeighted(
                self._current_image, 1, line_image, 0.5, 1
            )

        # display triangle used for lane and collision
        if self._DISPLAY_TRIANGLE:
            pts = np.array([self._np_lines], np.int32)
            cv2.polylines(self._current_image, pts, True, (0, 255, 255), 2)

    def _read(self):

        try:
            data = self.sock.recv(2048)
        except BlockingIOError:
            pass
        else:
            if data:
                if DEBUG:
                    logger.debug("Recieved data: ", len(data))
                self._recv_buffer += data
            else:
                logger.error("Peer closed")
                raise RuntimeError("Peer closed.")

    def read(self):
        """ Main function for reading and processing a message"""

        self._read()

        if self._read_header:
            self._get_header()
            self._process_header()

        self._process_request()

    def close(self):

        logger.info("Closing connection ...")

        try:
            self.selector.unregister(self.sock)
        except Exception as e:
            logger.exception(
                "error: selector.unregister() exception for", f"{self.addr}: {repr(e)}"
            )

        try:
            self.sock.close()
        except OSError as e:
            logger.exception(
                "error: socket.close() exception for", f"{self.addr}: {repr(e)}"
            )

        finally:
            # Delete reference to socket object for garbage collection
            self.sock = None
