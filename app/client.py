import socket
import select
import errno
import sys


HEADER_SIZE = 10
IP = "127.0.0.1"
PORT = 1234

my_username = input("Username: ")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))
client_socket.setblocking(False) # receive functionality will not be blocking

username = my_username.encode("utf-8") # convert username to bytes
username_header = f"{len(username):<{HEADER_SIZE}}".encode("utf-8")
client_socket.send(username_header + username)

while True:
    message = input(f"{my_username} > ")

    if message:
        message = message.encode("utf-8")
        message_header = f"{len(message):<{HEADER_SIZE}}".encode("utf-8")
        client_socket.send(message_header + message)

    try:
        while True:
            # receive
            username_header = client_socket.recv(HEADER_SIZE)
            if not len(username_header):
                print("connection closed by server")
                sys.exit()

            username_length = int(username_header.decode("utf-8").strip())
            username = client_socket.recv(username_length).decode("utf-8")

            message_header = client_socket.recv(HEADER_SIZE)
            message_length = int(message_header.decode("utf-8").strip())
            message = client_socket.recv(message_length).decode("utf-8")

            print(f"{username} > {message}")

    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('reading error', str(e))
            sys.exit()
        continue
    except:
        print('general error', str(e))
        sys.exit()







# import socket
# import pickle

# HEADER_SIZE = 10

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((socket.gethostname(), 1237))


# while True:
#     full_msg = b'' # b for bytes
#     new_msg = True
#     while True:
#         msg = s.recv(16) # more than header size to ensure the header is recvd
#         if new_msg:
#             print(f"new message length: {msg[:HEADER_SIZE]}")
#             msg_len = int(msg[:HEADER_SIZE]) # only header from the beginning to index 10
#             new_msg = False

#         # full_msg += msg.decode("utf-8")  # decode to convert from bytes to string
#         full_msg += msg  # don't need to decode as we want it bytes

#         if len(full_msg)-HEADER_SIZE == msg_len:
#             print("full msg recvd")
#             # print(full_msg[HEADER_SIZE:]) # print only msg from 10 to the end

#             d = pickle.loads(full_msg[HEADER_SIZE:]) # decode
#             print(d)

#             new_msg = True
#             full_msg = b''

#     print(full_msg)