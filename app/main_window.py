
# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from ui_main_window import *

import random

import socket
import select
import errno




user_type = 1   # server 1, client 0
username_from_gui = ""
PORT = 1235
HEADER_SIZE = 10
IP = "127.0.0.1"

client_socket = None


#=====================================================================================================================================================
#--------------------------------------------------------- WELCOME WINDOW ----------------------------------------------------------------------------
#=====================================================================================================================================================


class Who_Are_You(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        # Create widgets
        self.label = QtWidgets.QLabel("what do you want to do?")
        self.server = QtWidgets.QRadioButton("create new meeting")
        self.client = QtWidgets.QRadioButton("join meeting")
        self.button = QtWidgets.QPushButton("done")

        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
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
        # self.connect(self.button, QtCore.SIGNAL("clicked()"), qApp, QtCore.SLOT("quit()"))

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
#------------------------------------------------------- MAIN WINDOW SERVER --------------------------------------------------------------------------
#=====================================================================================================================================================


class MainWindowServer(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form_Server()
        self.ui.setupUi(self)
        self.ui.meeting_ID.setText(QtCore.QCoreApplication.translate("Form", str(PORT)))

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)
        self.initialize_server_socket()

    # initialize socket
    def initialize_server_socket(self):
        global IP
        global PORT
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((IP, PORT))
        self.server_socket.listen()
        self.sockets_list = [self.server_socket]
        self.clients = {}

    # receive message
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

    # accept new user trying to connect
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

    # receive message and send it to all
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

    # view camera
    def viewCam(self):   # loop here
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

        print("before hell")
        print(self.sockets_list)
        #----------------------------------------------------------
        read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)
        print("entering hell")
        for notified_socket in read_sockets:
            print("loop how?")
            if notified_socket == self.server_socket: # someone just connect, new user
                self.connect()

            else: # this is msg not new user
                self.send_to_all(notified_socket)

        # remove exeption sockets if any
        for notified_socket in exception_sockets:
            self.sockets_list.remove(notified_socket)
            del self.clients[notified_socket]

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")


if __name__ == '__main__':


    app = QApplication(sys.argv)
    widget = Who_Are_You()
    widget.show()
    app.exec_()

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    # create and show mainWindow
    mainWindow = MainWindowServer()
    mainWindow.show()

    sys.exit(app.exec_())