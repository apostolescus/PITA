import socket
import selectors
import traceback
import server_message

HOST = "127.0.0.1"
PORT = 65432

selector = selectors.DefaultSelector()

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen()
listen_socket.setblocking(False)
selector.register(listen_socket, selectors.EVENT_READ, data=None)

def add_connection(sock):

    conn, addr = sock.accept()
    conn.setblocking(False)
    message = server_message.Message(selector, conn, addr)
    selector.register(conn, selectors.EVENT_READ, message)
    print("Connection added")

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
 

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     #s.setblocking(False)
#     conn, addr = s.accept()

#     with conn:
#         print('Connected by: ', addr)
#         while True:
#             buffer = conn.recv()
#             print("I've recived some data")
#             if not data:
#                 break
#             print("Data", data)