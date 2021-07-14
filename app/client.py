import socket
import pickle

HEADER_SIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1237))


while True:
    full_msg = b'' # b for bytes
    new_msg = True
    while True:
        msg = s.recv(16) # more than header size to ensure the header is recvd
        if new_msg:
            print(f"new message length: {msg[:HEADER_SIZE]}")
            msg_len = int(msg[:HEADER_SIZE]) # only header from the beginning to index 10
            new_msg = False

        # full_msg += msg.decode("utf-8")  # decode to convert from bytes to string
        full_msg += msg  # don't need to decode as we want it bytes

        if len(full_msg)-HEADER_SIZE == msg_len:
            print("full msg recvd")
            # print(full_msg[HEADER_SIZE:]) # print only msg from 10 to the end

            d = pickle.loads(full_msg[HEADER_SIZE:]) # decode
            print(d)

            new_msg = True
            full_msg = b''

    print(full_msg)