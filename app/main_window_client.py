
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

from ui_main_window_client import *

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
#------------------------------------------------------- ENTER MEETING --------------------------------------------------------------------------
#=====================================================================================================================================================



class enter_user_name(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # Create widgets
        self.username_lbl = QtWidgets.QLabel("username:")
        self.username_txt = QtWidgets.QLineEdit()
        self.meeting_id_lbl = QtWidgets.QLabel("meeting ID:")
        self.meeting_id_txt = QtWidgets.QLineEdit()
        self.add_me = QtWidgets.QPushButton("Add me")
        self.confirm_info = QtWidgets.QLabel("")
        self.button = QtWidgets.QPushButton("done")
        self.button.hide()
        # Create layout and add widgets
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.username_lbl)
        layout.addWidget(self.username_txt)
        layout.addWidget(self.meeting_id_lbl)
        layout.addWidget(self.meeting_id_txt)
        layout.addWidget(self.add_me)
        layout.addWidget(self.confirm_info)
        layout.addWidget(self.button)
        # Set dialog layout
        self.setLayout(layout)

        self.add_me.clicked.connect(self.add_user)

        # self.connect(self.button, QtCore.SIGNAL("clicked()"), qApp, QtCore.SLOT("quit()"))

    def add_user(self):
        self.my_username = self.username_txt.text()
        global PORT
        global username_from_gui
        username_from_gui = self.my_username
        PORT = int(self.meeting_id_txt.text())
        confirm = f"username: {self.my_username}, meeting ID: {PORT}"
        self.confirm_info.setText(confirm)
        self.initialize_client_socket()
        self.button.show()

    def initialize_client_socket(self):
        global client_socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((IP, PORT))
        client_socket.setblocking(False) # receive functionality will not be blocking

        username = self.my_username.encode("utf-8") # convert username to bytes
        username_header = f"{len(username):<{HEADER_SIZE}}".encode("utf-8")
        client_socket.send(username_header + username)



#=====================================================================================================================================================
#------------------------------------------------------- MAIN WINDOW CLIENT --------------------------------------------------------------------------
#=====================================================================================================================================================


class MainWindowClient(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form_Client()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)


        global client_socket
        self.client_socket = client_socket

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
        #----------------------------------------------------------
        message = "hello"
        if message:
            message = message.encode("utf-8")
            message_header = f"{len(message):<{HEADER_SIZE}}".encode("utf-8")
            self.client_socket.send(message_header + message)               # send message


        #-----------------------------------------------------------------------------
        try:
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
        except:
            print('general error', str(e))
            sys.exit()

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture("04.mp4")
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
    mainWindow = enter_user_name()
    mainWindow.show()
    app.exec_()

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    # create and show mainWindow
    mainWindow = MainWindowClient()
    mainWindow.show()

    sys.exit(app.exec_())