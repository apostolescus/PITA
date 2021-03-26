import sys
import socket
import selectors
import traceback
import cv2
import base64

import client_message

HOST = "127.0.0.1"
PORT = 65432

sel = selectors.DefaultSelector()

def generate_request():

    image = cv2.imread("test.jpeg")
    encoded_image = base64.b64encode(cv2.imencode('.jpg', image)[1])
    mode = "detect"
    
    return dict(mode=mode, content=encoded_image)

action = generate_request()

def start_connection(request):
    addr = (HOST, PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = client_message.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)

start_connection(action)

try:
    while True:
        events = sel.select(timeout=1)
        for key, mask in events:
            message = key.data
            try:
                message.process_message(mask)
            except Exception:
                print(
                    "main: error: exception for",
                    f"{message.addr}:\n{traceback.format_exc()}",
                )
                message.close()
except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    sel.close()
