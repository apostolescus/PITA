"""This module should be run to start the PITA program from the client side.
It will open a connection with the server and send a message.
Modifiy HOST and PORT to your server.
If you want to use localhost for testing put HOST='127.0.0.1'.
Don't forget to modify in the sever side script too."""

import socket
import selectors
import traceback
import ssl

from screen_manager import GUIManagerThread
from storage import toggle_update_message, logger, config_file
from gps import GPS

import client_message

HOST = config_file["SERVER"]["ip"]
PORT = config_file["SERVER"].getint("port")

sel = selectors.DefaultSelector()


def start_connection() -> None:
    """Initialize socket connection with server.
    Loads and verifies certificates."""

    server_sni_hostname = config_file["CERTIFICATES"]["hostname"]
    server_cert = config_file["CERTIFICATES"]["server_cert"]
    client_cert = config_file["CERTIFICATES"]["client_cert"]
    client_key = config_file["CERTIFICATES"]["client_key"]

    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=server_cert)
    context.load_cert_chain(certfile=client_cert, keyfile=client_key)

    toggle_update_message()
    addr = (HOST, PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    conn = context.wrap_socket(
        sock, server_side=False, server_hostname=server_sni_hostname
    )
    conn.connect(addr)

    event = selectors.EVENT_READ | selectors.EVENT_WRITE
    start_message = client_message.Message(sel, conn, addr)
    sel.register(conn, event, data=start_message)


start_connection()

try:
    # start the GUI
    guiManager = GUIManagerThread("guiThread")
    guiManager.start()

    # start GPS thread
    current_gps = GPS("gps-thread")
    current_gps.start()

    while True:
        events = sel.select(timeout=1)
        for key, mask in events:
            message = key.data
            try:
                message.process_message(mask)
            except Exception:
                logger.exception(
                    "main: error: exception for",
                    f"{message.addr}:\n{traceback.format_exc()}",
                )
                message.close()

except KeyboardInterrupt:
    logger.exception("caught keyboard interrupt, exiting ... ")
finally:
    sel.close()
