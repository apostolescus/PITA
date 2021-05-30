"""Module responsible for handling communication messages between server and client.
"""
import time
import io
import base64
from threading import Thread

import selectors
import json
import struct
import numpy as np
import cv2
import notify2
from playsound import playsound

# import locals
from screen_manager import captured_image_queue, result_queue
from storage import toggle_update_message, get_update_message, config_file
from storage import UISelected, get_switch_sound, get_gps_infos, logger

# globals
lane_detection = False
first_message = False

# notification initialization
notify2.init('PITA')

# debugg
counter = config_file["DEBUG"].getint("video_frames")
DEBUG = bool(config_file["DEBUG"].getboolean("verbose"))
TIME = bool(config_file["DEBUG"].getboolean("time"))
average_time = 0
average_time_counter = 0


def play_sound():
    playsound("alert_sounds/beep.mp3")


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
                logger.log(
                    leve="DEBUG",
                    message="---main thread--- client switched to listen mode",
                )
            events = selectors.EVENT_READ

        elif mode == "w":
            if DEBUG:
                logger.log(
                    leve="DEBUG",
                    message="---main thread--- client switched to write mode",
                )
            events = selectors.EVENT_WRITE

        elif mode == "rw":
            if DEBUG:
                logger.log(
                    leve="DEBUG",
                    message="---main thread--- client switched to read/write mode",
                )
            events = selectors.EVENT_READ | selectors.EVENT_WRITE

        else:
            logger.exception(f"Invalid events mask mode {repr(mode)}.")

        self.selector.modify(self.sock, events, data=self)

    def _generate_request(self):
        """Method that generates a request for the server."""

        msg: str = ""

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
                }
                UISelected.updated = False
            else:
                header = {
                    "request-type": "UPDATE",
                    "update": 0,
                    "time": time.time(),
                    "lane": UISelected.lane_detection,
                }

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
            gps_infos = get_gps_infos()

            gps_speed = gps_infos[0]
            gps_lat = gps_infos[1]
            gps_lon = gps_infos[2]

            # create header dictionary
            header = {
                "request-type": "DETECT",
                "time": time.time(),
                "speed": gps_speed,
                "lat": gps_lat,
                "lon": gps_lon,
                # "uuid":str(uuid.uuid4()),
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

        if DEBUG:
            logger.debug("--- main_thread --- processing request")

        if json_type == "EMPTY":

            if DEBUG:
                logger.debug("--- main_thread --- empty request")

            self._read_header = True
            self._request_done = False

            self._set_selector_events_mask("w")

        elif json_type == "DETECTED":
            global average_time, average_time_counter

            if DEBUG:
                logger.debug("--- main_thread --- detected request")

            self._request_done = False
            content_len = self.json_header["content-len"]

            if TIME:
                send_time = self.json_header["time"]
                dif = time.time() - send_time
                if average_time_counter > 100:
                    logger.info(
                        "average time is: ", average_time / average_time_counter
                    )

                else:
                    average_time += dif
                    average_time_counter += 1
                logger.info("From server to client: ", str(dif))

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
        alerts = response["alerts"]

        # display alerts
        if alerts:
            for alert in alerts:
                n = notify2.Notification('ALERT', alert)
                n.show()
                time.sleep(0.2)

        switch_sound = get_switch_sound()
        if danger == 1 and switch_sound:
            
            t = Thread(target=play_sound)
            t.start()

        height = self._current_image.shape[0]
        width = self._current_image.shape[1]

        pts = np.array(
            [
                [
                    (width - 530, height - 50),
                    (width / 2 - 15, 200),
                    (width - 120, height - 50),
                ]
            ],
            np.int32,
        )

        image = self._current_image

        if obj_list is not None:
            for obj in obj_list:

                x, y = obj["coordinates"][0], obj["coordinates"][1]
                w, h = obj["coordinates"][2], obj["coordinates"][3]

                color = obj["color"]
                label = obj["label"]
                score = obj["score"]
                text = "{}: {:.4f}".format(label, score)

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

            # print("Lines: ", type(line))
            # self._current_image = cv2.addWeighted(self._current_image, 1, line, 0.5, 1)

        # only for experimental/ debugging purpose
        if counter > 0:
            self.out.write(self._current_image)
            counter -= 1
        else:
            if self.saved is False:
                print("VIDEO SAVED SUCCESFULLY")
                self.out.release()
                self.saved = True

        # display triangle used for lane and collision
        # cv2.polylines(self._current_image, pts, True, (0, 255, 255), 2)

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
