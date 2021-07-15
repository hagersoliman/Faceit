import sys
from PySide2.QtWidgets import *
from PySide2 import *

user_type = 0
username = "sara"
PORT = 1234

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
        global user_type
        user_type = 1
        global PORT
        PORT = 1234

    def set_type_client(self):
        global user_type
        user_type = 0
        

class enter_user_name(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # Create widgets
        self.username_lbl = QLabel("username:")
        self.username_txt = QLineEdit()
        self.meeting_id_lbl = QLabel("meeting ID:")
        self.meeting_id_txt = QLineEdit()
        self.add_me = QPushButton("Add me")
        self.confirm_info = QLabel("")
        self.button = QPushButton("done")
        self.button.hide()
        # Create layout and add widgets
        layout = QVBoxLayout()
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

        self.connect(self.button, QtCore.SIGNAL("clicked()"), qApp, QtCore.SLOT("quit()"))

    def add_user(self):
        global username
        username = self.username_txt.text()
        global PORT
        PORT = int(self.meeting_id_txt.text())
        confirm = f"username: {username}, meeting ID: {PORT}"
        self.confirm_info.setText(confirm)
        self.button.show()



class create_client(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # Create widgets
        self.client_name = QLabel("client")
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        self.type_msg = QLineEdit()
        self.send_msg = QPushButton("send")
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.client_name)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.type_msg)
        layout.addWidget(self.send_msg)
        print("client")
        # Set dialog layout
        self.setLayout(layout)


class create_server(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        # Create widgets
        self.server_name = QLabel("server")
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.server_name)
        layout.addWidget(self.textEdit)
        print("server")
        # Set dialog layout
        self.setLayout(layout)


app = QtWidgets.QApplication(sys.argv)
widget = Who_Are_You()
widget.show()
# sys.exit(app.exec_())
app.exec_()

if user_type == 1:
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    widget = create_server()
    widget.show()
    sys.exit(app.exec_())

elif user_type == 0:
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    widget = enter_user_name()
    widget.show()
    app.exec_()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    widget = create_client()
    widget.show()
    sys.exit(app.exec_())





