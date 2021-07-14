import socket
import time
import pickle

HEADER_SIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1237))
s.listen(5)

while True:
    client_socket, address = s.accept()
    print(f"connect feom {address} has been established")

    d = {1: "hello", 2: "world"}
    msg = pickle.dumps(d) # convert d to bytes

    msg = bytes(f'{len(msg):<{HEADER_SIZE}}', "utf-8") + msg # convert header to bytes
    client_socket.send(msg)

    # the folloing lines to test time
    # while True:
    #     time.sleep(3)
    #     msg = f"the time is {time.time()}"
    #     msg = f'{len(msg):<{HEADER_SIZE}}' + msg
    #     client_socket.send(bytes(msg, "utf-8"))
