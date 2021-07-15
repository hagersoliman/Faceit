import socket
import select
import errno

import sys

from PySide2.QtWidgets import *
from PySide2 import *

import random

#============================================================================================
#-------------------------------------------- GUI -------------------------------------------
#============================================================================================

user_type = 1   # server 1, client 0
username_from_gui = ""
PORT = 1235
HEADER_SIZE = 10
IP = "127.0.0.1"

class Who_Are_You(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # Create widgets
        self.label = QLabel("what do you want to do?")
        self.server = QRadioButton("create new meeting")
        self.client = QRadioButton("join meeting")
        self.button = QPushButton("done")

        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.server)
        layout.addWidget(self.client)
        layout.addWidget(self.button)

        # Set dialog layout
        self.setLayout(layout)
        global user_type
        print("who are you")

        self.server.clicked.connect(self.set_type_server)
        self.client.clicked.connect(self.set_type_client)
        self.connect(self.button, QtCore.SIGNAL("clicked()"), qApp, QtCore.SLOT("quit()"))

    def set_type_server(self):
        print("server choosen")
        global user_type
        user_type = 1
        global PORT
        PORT = random.randrange(1234, 9876, 1)

    def set_type_client(self):
        print("client choosen")
        global user_type
        user_type = 0
   

#=====================================================================================================================================================
#------------------------------------------------------------ SERVER ---------------------------------------------------------------------------------
#=====================================================================================================================================================


class create_server(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # Create widgets
        self.server_name = QLabel("server")
        self.meeting_id_lbl = QLabel("")
        self.meeting_id_lbl.setText(str(PORT))
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.server_name)
        layout.addWidget(self.meeting_id_lbl)
        layout.addWidget(self.textEdit)
        print("server")
        # Set dialog layout
        self.setLayout(layout)
        self.initialize_server_socket()

    def initialize_server_socket(self):
        global IP
        global PORT
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((IP, PORT))
        self.server_socket.listen()
        self.sockets_list = [self.server_socket]
        self.clients = {}

    def receive_message(self, client_socket):
        global HEADER_SIZE
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

    def connect(self):
        #accept this connection
        client_socket, client_address = self.server_socket.accept()
        # receive user's data 
        user = self.receive_message(client_socket)
        if user is False:  # user disconnected
            return

        self.sockets_list.append(client_socket) # add this client socket to socket list
        self.clients[client_socket] = user # add this data {username} to clients
        print(f"accepted new connection from {client_address[0]}:{client_address[1]} username:{user['data'].decode('utf-8')}")

    def send_to_all(self, notified_socket):
        message = self.receive_message(notified_socket) 

        if message is False: # disconnected
            print(f"closed connection from {self.clients[notified_socket]['data'].decode('utf-8')}")
            self.sockets_list.remove(notified_socket)
            del self.clients[notified_socket]
            return

        user = self.clients[notified_socket]
        print(f"received msg from {user['data'].decode('utf-8')}: {message['data'].decode('utf-8')}")

        for client_socket in self.clients:
            if client_socket != notified_socket: # send to all exept the sender
                # send username then msg
                client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

    def run_server(self):
        while True:
            read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

            for notified_socket in read_sockets:
                if notified_socket == self.server_socket: # someone just connect, new user
                    self.connect()

                else: # this is msg not new user
                    self.send_to_all(notified_socket)

            # remove exeption sockets if any
            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)
                del self.clients[notified_socket]


#=====================================================================================================================================================

app = QtWidgets.QApplication(sys.argv)
widget = Who_Are_You()
widget.show()
app.exec_()

# if server
if user_type == 1:
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    widget = create_server()
    widget.show()
    sys.exit(app.exec_())
#=====================================================================================================================================================
#----------------------------------------------------------------------- SOCKET ----------------------------------------------------------------------
#=====================================================================================================================================================

# user_type = input("client or server? (c/s) ")


HEADER_SIZE = 10
IP = "127.0.0.1"
# PORT = 1235


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# if server
if user_type == 1: 
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
if user_type == 0: 
    # TODO connect with input port
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
