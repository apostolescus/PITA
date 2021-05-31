"""This module should be run to start the PITA program from the server side.
It will listen on the specified port for a connection and will check the client certification.
Modifiy HOST and PORT to your server ip.
If you want to use localhost for testing put HOST='127.0.0.1'.
Don't forget to modify in the client side script too."""

import socket
import selectors
import traceback
import ssl

import server_message
from storage import config_file, logger

HOST = config_file["SERVER"]["ip"]
PORT = config_file["SERVER"].getint("port")

selector = selectors.DefaultSelector()

# declaring certificates locations
server_cert = config_file["CERTIFICATES"]["server_cert"]
server_key = config_file["CERTIFICATES"]["server_key"]
client_certs = config_file["CERTIFICATES"]["client_cert"]

# initialize listening socket
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# bind it to host and port
listen_socket.bind((HOST, PORT))

# set to non-blockin and start listening mode
listen_socket.setblocking(False)
listen_socket.listen()

selector.register(listen_socket, selectors.EVENT_READ, data=None)


def add_connection(sock):

    # initialize and request client authentication
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.verify_mode = ssl.CERT_REQUIRED

    # load certificates
    context.load_cert_chain(certfile=server_cert, keyfile=server_key)

    # load client verification certificate
    context.load_verify_locations(cafile=client_certs)

    # accept connection
    newsocket, addr = sock.accept()

    # wrap in secure connection
    conn = context.wrap_socket(newsocket, server_side=True)

    # register connection
    message = server_message.Message(selector, conn, addr)
    selector.register(conn, selectors.EVENT_READ, message)


try:
    while True:
        event = selector.select(timeout=None)
        for key, mask in event:
            if key.data is None:
                add_connection(key.fileobj)
            else:
                message = key.data
                try:
                    message.process_message(mask)
                except Exception:
                    logger.exception(
                        "main: error: exception for" +
                        str(message.addr)
                    )
                    message.close()

except KeyboardInterrupt:
    logger.level("SERVER", "Keyboard interrupting, exiting ...")
finally:
    selector.close()
