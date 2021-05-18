"""This module should be run to start the PITA program from the client side.
It will open a connection with the server and send a message.
Modifiy HOST and PORT to your server.
If you want to use localhost for testing put HOST='127.0.0.1'.
Don't forget to modify in the sever side script too."""

import socket
import selectors
import traceback

from screen_manager import GUIManagerThread
from storage import toggle_update_message

import client_message

# HOST = "194.61.21.139"
HOST = "127.0.0.1"
PORT = 65432

sel = selectors.DefaultSelector()


def start_connection():
    """ Initialize connection with the server"""
    toggle_update_message()
    addr = (HOST, PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    event = selectors.EVENT_READ | selectors.EVENT_WRITE
    start_message = client_message.Message(sel, sock, addr)
    sel.register(sock, event, data=start_message)


start_connection()

try:
    # start the GUI
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
