import sys
import socket
import selectors
import traceback
import cv2
import base64

from screen_manager import GUIManagerThread
from screen_manager import captured_image_queue, result_queue
from storage import toggle_update_message

import client_message

HOST = "194.61.21.75"
#HOST = "127.0.0.1"
PORT = 65432

sel = selectors.DefaultSelector()


def start_connection():
    toggle_update_message()
    addr = (HOST, PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = client_message.Message(sel, sock, addr)
    sel.register(sock, events, data=message)


start_connection()

try:
    guiManager = GUIManagerThread("guiThread")
    guiManager.start()

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
