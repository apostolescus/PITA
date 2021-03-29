import selectors
import json
import sys
import struct
import io

import cv2
import base64
import uuid

#import locals
from screen_manager import captured_image_queue, result_queue
from Storage import toggle_update_message, get_update_message
from Storage import UISelected

#debugg
counter = 0
max_val = 10000
mute = True

    
class Message:
    def __init__(self, selector, sock, addr, request):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._request = request
        self._request_done = False
        self._read_header = True
        self._send_buffer = b""
        self._recv_buffer = b""
        self._current_image = None

    def process_message(self, mask):
        if mask & selectors.EVENT_WRITE:
            self.write() 
        if mask & selectors.EVENT_READ:
            self.read()

    def _json_decode(self, json_bytes, encoding="utf-8"):
        
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes), encoding=encoding, newline=""
        )
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

        msg = ""
        if get_update_message():
            # send update message
            toggle_update_message()

            header = {
                "request-type" : "UPDATE",
                "lane" : UISelected.lane_detection,
                "car_type" : UISelected.car_type,
                "weather" : UISelected.weather,
                "experience" : UISelected.experience,
                "rec_mode" : UISelected.rec_mode,
                "reaction_time" : UISelected.reaction_time
            }

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header

        else:
            self._current_image = captured_image_queue.get()
            content = base64.b64encode(cv2.imencode('.jpg', self._current_image)[1])
            
            header = {
                    "request-type":"DETECT",
                    "content-len":len(content),
                    "encode-type":"base64",
                    "request-id":str(uuid.uuid4())
                    }
            if not mute:
                print("Request id: ", header["request-id"])

            encoded_header = self._json_encode(header)
            message_hdr = struct.pack("<H", len(encoded_header))
            msg = message_hdr + encoded_header + content

        self._send_buffer += msg
        self._request_done = True

        return msg

    def write(self):
        global counter

        if counter == max_val:
            self.close()
            return
        counter+=1

        if self._request_done is False:
            self._generate_request()
        
        self._write()

        if self._request_done:
            if not self._send_buffer:
                self._set_selector_events_mask("r")

    def _write(self):
        
        if self._send_buffer:
            try:
                sent = self.sock.send(self._send_buffer)
                if not mute:
                    print("--- main thread --- Sended: ", sent)
            except BlockingIOError:
                pass
            else:
                self._send_buffer = self._send_buffer[sent:]

    def _get_header(self):
        header_len = 2

        if len(self._recv_buffer) >=  header_len:
            self._header_len = struct.unpack("<H", self._recv_buffer[:header_len])[0]
            self._recv_buffer = self._recv_buffer[header_len:]

    def _process_header(self):
        header_len = self._header_len

        if len(self._recv_buffer) >= header_len:
            self.json_header = self._json_decode(self._recv_buffer[:header_len])
            #print("--json header --- ", self.json_header)
            self._recv_buffer = self._recv_buffer[header_len:]
            # print("Json header: ", self.json_header)

    def _process_request(self):
        
        json_type = self.json_header["response-type"]

        if not mute:
            print("JSON type", json_type)

        if json_type == "EMPTY":
            self._read_header = True
            self._request_done  = False
            self.write()
            
            #send another frame
        elif json_type == "DETECTED":
            self._request_done  = False
            content_len = self.json_header["content-len"]
            if len(self._recv_buffer) >= content_len:
                self._read_header = True
                data = self._recv_buffer[:content_len]
                self._recv_buffer = self._recv_buffer[content_len:]
                #print("String is: ", str(data))
                decoded_response = self._json_decode(data)

                self._display_image(decoded_response)
                # print("Response is: ", decoded_response)
                self._set_selector_events_mask("w")

        result_queue.put(self._current_image)
        # cv2.imshow("readimg", self._current_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #         return

    def _display_image(self, response):    

        
        # decoded_json = json.loads(decoded_response)

        obj_list = response["detected_objects"]

        for obj in obj_list:
            x,y  = obj["coordinates"][0], obj["coordinates"][1] 
            w,h = obj["coordinates"][2], obj["coordinates"][3]

            color = obj["color"]
            label = obj["label"]
            score = obj["score"]
            text = "{}: {:.4f}".format(label, score)
            image = self._current_image

            cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self._current_image = image

    def _read(self):

        try:
            data = self.sock.recv(1024)
        except BlockingIOError:
            pass
        else:
            if data:
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