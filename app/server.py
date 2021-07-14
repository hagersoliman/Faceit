import socket
import time

HEADER_SIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1236))
s.listen(5)

while True:
    client_socket, address = s.accept()
    print(f"connect feom {address} has been established")

    msg = "welcome to the server"
    msg = f'{len(msg):<{HEADER_SIZE}}' + msg
    client_socket.send(bytes(msg, "utf-8"))

    # the folloing lines to test time
    while True:
        time.sleep(3)
        msg = f"the time is {time.time()}"
        msg = f'{len(msg):<{HEADER_SIZE}}' + msg
        client_socket.send(bytes(msg, "utf-8"))
