import socket
import selectors
import traceback
import ssl

import server_message

HOST = "192.168.10.39"
PORT = 65432

selector = selectors.DefaultSelector()

# declaring certificates locations
server_cert = 'server_certificate/server.crt'
server_key = 'server_certificate/server.key'
client_certs = 'client_certificates/client.crt'

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

    #newsocket.setblocking(False)

    # wrap in secure connection
    conn = context.wrap_socket(newsocket, server_side=True)

    # register connection
    #conn.do_handshake()

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
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()

except KeyboardInterrupt:
    print("caught keyboard interrupt, exiting")
finally:
    selector.close()
