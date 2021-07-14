import socket
import select

HEADER_SIZE = 10
IP = "127.0.0.1"
PORT = 1234

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((IP, PORT))
server_socket.listen()

sockets_list = [server_socket]

clients = {}

# server receive messages from clients
def receive_message(client_socket):
    try:
        # receive message header only
        message_header = client_socket.recv(HEADER_SIZE)

        # if there is no msg return false
        if not len(message_header):
            return False

        message_length = int(message_header.decode("utf-8").strip())
        msg_data = client_socket.recv(message_length) # get data, here all data reveived at once, we can devide it as before
        return {"header": message_header, "data": msg_data} #return header and data

    except:
        return False

while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)

    for notified_socket in read_sockets:
        if notified_socket == server_socket: # someone just connect, new user
            #accept this connection
            client_socket, client_address = server_socket.accept()
            # receive user's data 
            user = receive_message(client_socket)
            if user is False:  # user disconnected
                continue

            sockets_list.append(client_socket) # add this client socket to socket list
            clients[client_socket] = user # add this data {username} to clients

            print(f"accepted new connection from {client_address[0]}:{client_address[1]} username:{user['data'].decode('utf-8')}")

        else:   # this is msg not new user
            # get that msg
            message = receive_message(notified_socket) 

            if message is False: # disconnected
                print(f"closed connection from {clients[notified_socket]['data'].decode('utf-8')}")
                sockets_list.remove(notified_socket)
                del clients[notified_socket]
                continue

            user = clients[notified_socket]
            print(f"received msg from {user['data'].decode('utf-8')}: {message['data'].decode('utf-8')}")

            for client_socket in clients:
                if client_socket != notified_socket: # send to all exept the sender
                    # send username then msg
                    client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

    # remove exeption sockets if any
    for notified_socket in exception_sockets:
        sockets_list.remove(notified_socket)
        del clients[notified_socket]










# import socket
# import time
# import pickle

# HEADER_SIZE = 10

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((socket.gethostname(), 1237))
# s.listen(5)

# while True:
#     client_socket, address = s.accept()
#     print(f"connect feom {address} has been established")

#     d = {1: "hello", 2: "world"}
#     msg = pickle.dumps(d) # convert d to bytes

#     msg = bytes(f'{len(msg):<{HEADER_SIZE}}', "utf-8") + msg # convert header to bytes
#     client_socket.send(msg)

#     # the folloing lines to test time
#     # while True:
#     #     time.sleep(3)
#     #     msg = f"the time is {time.time()}"
#     #     msg = f'{len(msg):<{HEADER_SIZE}}' + msg
#     #     client_socket.send(bytes(msg, "utf-8"))
