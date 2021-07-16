import socket
import select
import errno
import sys
import random
import time

user_type = input("client or server? (c/s) ")


HEADER_SIZE = 10
IP = "127.0.0.1"
PORT = 1239

# if server
if user_type == 's': 
    # TODO generate randon port
    # print the port
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


# if client
if user_type == 'c': 
    # TODO connect with input port
    my_username = input("Username: ")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((IP, PORT))
    client_socket.setblocking(False) # receive functionality will not be blocking

    username = my_username.encode("utf-8") # convert username to bytes
    username_header = f"{len(username):<{HEADER_SIZE}}".encode("utf-8")
    client_socket.send(username_header + username)

    while True:
        time.sleep(2)
        message = str(random.randrange(1234, 9876, 1))
        # message = input(f"{my_username} > ")
        # print("one")

        if message:
            # print("two")
            message = message.encode("utf-8")
            message_header = f"{len(message):<{HEADER_SIZE}}".encode("utf-8")
            client_socket.send(message_header + message)

        try:
            while True:
                # print("before")
                # receive
                username_header = client_socket.recv(HEADER_SIZE)
                # print("after")
                if not len(username_header):
                    print("connection closed by server")
                    sys.exit()

                # print("three")
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
