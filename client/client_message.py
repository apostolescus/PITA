"""Module responsible for handling communication messages between server and client.
"""
import time
import base64
from threading import Thread

import selectors
import json
import struct
import numpy as np
import cv2
import io

from playsound import playsound
from gps import GPS

# import locals
from screen_manager import captured_image_queue, result_queue
from storage import toggle_update_message, get_update_message
from storage import UISelected, get_switch_sound, get_gps_infos

#globals 
lane_detection = False

# debugg
counter = 0
start = False
max_val = 10000
mute = True
timing = True
average_time = 0
average_time_counter = 0

class Message:
    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._request_done = False
        self._read_header = True
        self._send_buffer = b""
        self._recv_buffer = b""
        self._current_image = None
        
        #only for lane detection comparision
        self.out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 6, (640, 480))
        self.counter = 65
        self.saved = False

    def process_message(self, mask):
        if mask & selectors.EVENT_WRITE:
            self.write()
        if mask & selectors.EVENT_READ:
            self.read()

    def _json_decode(self, json_bytes, encoding="utf-8"):

        tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
        obj = json.load(tiow)
        tiow.close()
        return obj

    def _json_encode(self, stream):
        return json.dumps(stream, ensure_ascii=False).encode()

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            if not mute:
                print("---main thread--- client switched to listen mode")
            events = selectors.EVENT_READ
        elif mode == "w":
            if not mute:
                print("---main thread--- client switched to write mode")
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            if not mute:
                print("---main thread--- client switched to read/write mode")
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _generate_request(self):
        global start

        msg = ""
        if mute is False:
            print("--- generate_request --- entered in generate request")

        if get_update_message():
            global lane_detection
            # send update message
            toggle_update_message()
            if mute is False:
                print("--- generate_request --- update message")
            if UISelected.updated:

                header = {
                    "request-type": "UPDATE",
                    "update": 1,
                    #"lane": UISelected.lane_detection,
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
                    "time":time.time(),
                    "lane": UISelected.lane_detection,
                }
            lane_detection = UISelected.lane_detection
            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header

        else:
            # extract iamge from queue
            self._current_image = captured_image_queue.get()
            
            # for DEBUG only
            # if start is False:
            #     height = self._current_image.shape[0]
            #     width = self._current_image.shape[1]

            #     print("Polygon coordinates: ")
            #     print(width - 530, height - (height - 50))
            #     print(width / 2 - 15, height - (200))
            #     print(width - 120, height - (height - 50))
            #     start = True

            # encode image in base64
            content = base64.b64encode(cv2.imencode(".jpg", self._current_image)[1])

            # DEBUG
            if mute is False:
                print("--- generate_request --- content len: ", len(content))

            # get GPS infos
            gps_infos = get_gps_infos()

            gps_speed = gps_infos[0]
            gps_lat = gps_infos[1]
            gps_lon = gps_infos[2]

            # create header dictionary
            header = {
                "request-type": "DETECT",
                "time":time.time(),
                "speed":gps_speed,
                "lat":gps_lat,
                "lon":gps_lon,
                #"uuid":str(uuid.uuid4()),
                "content-len":len(content),
            }
            

            # if not mute:
            #     print("Request id: ", header["request-id"])

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header + content

        self._send_buffer += msg
        self._request_done = True

        return msg

    def write(self):
   
        if mute is False:
            print("--- write --- entering write function")

        if self._request_done is False:
            if mute is False:
                print("--- write --- request done is False")
            self._generate_request()

        # write all bufffer
        while True:
            if self._write():
                break

        if not mute:
            print("--- write --- processing request")

        if self._request_done:
            if not mute:
                print("--- write --- request done")
            if not self._send_buffer:
                if not mute:
                    print("--- write --- send buffer is empty")
                self._set_selector_events_mask("r")
            # else:
            #     print("Send buffer", self._send_buffer)

    def _write(self)->bool:

        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
        
            except BlockingIOError:
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]

         # if the buffer is not empty repeat sending
       
        if not self._send_buffer:
            return True
        else:
            return False

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

    def _process_request(self):

        json_type = self.json_header["response-type"]

        if not mute:
            print("--- main thread --- processing request")

        if json_type == "EMPTY":
            
            if not mute:
                print("--- main thread --- empty request")
                
            self._read_header = True
            self._request_done = False
            #self.write()
            self._set_selector_events_mask("w")
            

            # send another frame
        elif json_type == "DETECTED":
            global average_time, average_time_counter
            self._request_done = False
            content_len = self.json_header["content-len"]
            
            if timing:
                
                send_time = self.json_header["time"]
                dif = time.time() - send_time
                if average_time_counter > 100:
                    print("Average time is: ", average_time/ average_time_counter)
                else:
                    average_time += dif
                    average_time_counter += 1
                print("From server to client: ", str(dif))

            if not mute:
                print("--- main thread --- detected request")
            
            if len(self._recv_buffer) >= content_len:

                data = self._recv_buffer[:content_len]

                self._recv_buffer = self._recv_buffer[content_len:]
                self._read_header = True

                decoded_response = self._json_decode(data)
                
                self._display_image(decoded_response)
                
                self._set_selector_events_mask("w")

        result_queue.put(self._current_image)

    def _display_image(self, response):

        obj_list = response["detected_objects"]
        danger = response["danger"]
        line = response["lines"]

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
                print("OBJ IS: ", obj)
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
                    cv2.line(line_image,(x1, y1),(x2, y2),(0, 190, 255), 11)
                except:
                    return None
            self._current_image = cv2.addWeighted(self._current_image, 1, line_image, 0.5, 1)

       
            #print("Lines: ", type(line))
            #self._current_image = cv2.addWeighted(self._current_image, 1, line, 0.5, 1)

        #only for experimental/ debugging purpose
        if self.counter >= 0:    
            self.out.write(self._current_image)
            self.counter -= 1
        else:
            if self.saved is False:
                print("VIDEO SAVED SUCCESFULLY")
                self.out.release()
                self.saved = True

        #cv2.polylines(self._current_image, pts, True, (0, 255, 255), 2)

    def _read(self):

        try:
            data = self.sock.recv(2048)
        except BlockingIOError:
            pass
        else:
            if data:
                if not mute:
                    print("Recived data: ", len(data))
                self._recv_buffer += data
            else:
                raise RuntimeError("Peer closed.")

    def read(self):

        self._read()

        if self._read_header:
            self._get_header()
            self._process_header()

        self._process_request()

    def close(self):
        print("closing connection to", self.addr)
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

def play_sound():
    playsound("alert_sounds/beep.mp3")