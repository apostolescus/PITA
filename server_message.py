import selectors
import struct
import io
import json
import base64
import cv2
import numpy as np

class Message:
    def __init__(self, selector, sock, addr):
        self.selector = selector
        self.sock = sock
        self.addr = addr
        self._recv_buffer = b""
        self._send_buffer = b""
        self._switch = False
        self._read_header = True
        self._request_done = False

    def process_message(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
        if mask & selectors.EVENT_WRITE:
            self.write()

    def _set_selector_events_mask(self, mode):
        """Set selector to listen for events: mode is 'r', 'w', or 'rw'."""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _json_decode(self, json_bytes, encoding="utf-8"):
        
        tiow = io.TextIOWrapper(
            io.BytesIO(json_bytes), encoding=encoding, newline=""
        )
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
            #decoding header JSON
            self.json_header = self._json_decode(self._recv_buffer[:header_len])

            self._recv_buffer = self._recv_buffer[header_len:]
            self._read_header = False
    
    def _process_request(self):
        
        content_len = self.json_header["content-len"]

        if len(self._recv_buffer) >= content_len:

            data = self._recv_buffer[:content_len]
            self._recv_buffer = self._recv_buffer[content_len:]

            jpg_original = base64.b64decode(data)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)

            #process image

            # cv2.imwrite('0.jpg', img)
            
            self._switch = True
            self._read_header = True

    def read(self):
        self._read()
        
        if self._read_header:
            self._get_header()

            self._process_header()

        self._process_request()

        #check if queue is empty, else wait for other image
        if self._switch:
            self._set_selector_events_mask("w")

    def _generate_request(self):
        
        # add JSON with detected bbox and line
        # and encode it
        box_list = "raspuns cored"
        encoded_response = str.encode(box_list)

        # add custom header
        header = {
                'danger':0,
                'response-type':'DETECTION',
                'content-len':len(encoded_response)
                }

        encoded_header = self._json_encode(header)
        message_hdr = struct.pack("<H", len(encoded_header))

        msg = message_hdr + encoded_header + encoded_response

        self._send_buffer += msg
        self._request_done = True

    def _write(self):

        if self._switch:
            if self._send_buffer:
                try:
                    sent = self.sock.send(self._send_buffer)
                    self._request_done = False
                except BlockingIOError:
                    pass
                else:
                    self._send_buffer = self._send_buffer[sent:]  

        else:
            print("Switch is not true")

    def write(self):

        if self._request_done is False:
            self._generate_request()

        self._write()

        if not self._send_buffer:
            self._set_selector_events_mask("r")
            
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