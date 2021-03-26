import selectors
import json
import sys
import struct
import io

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
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        elif mode == "rw":
            events = selectors.EVENT_READ | selectors.EVENT_WRITE
        else:
            raise ValueError(f"Invalid events mask mode {repr(mode)}.")
        self.selector.modify(self.sock, events, data=self)

    def _generate_request(self):
        
        content = self._request["content"]
        mode = self._request["mode"]

        if mode == "detect":
            header = {
                    "request-type":"DETECT",
                    "content-len":len(content),
                    "encode-type":"base64"
                    }

            encoded_header = self._json_encode(header)
            # print("Encoded header: ", encoded_header)
            message_hdr = struct.pack("<H", len(encoded_header))
            # print("Header len is: ", len(encoded_header))

            msg = message_hdr + encoded_header + content

            self._send_buffer += msg
            self._request_done = True

        return msg

    def write(self):

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
            self._recv_buffer = self._recv_buffer[header_len:]
            # print("Json header: ", self.json_header)

    def _process_request(self):
        content_len = self.json_header["content-len"]

        if len(self._recv_buffer) >= content_len:
            data = self._recv_buffer[:content_len]
            print("String is: ", str(data))

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