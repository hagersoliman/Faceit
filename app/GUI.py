import sys
from PySide2.QtWidgets import *
from PySide2 import *

user_type = 0
username = "sara"

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
        self.label = QLabel("username:")
        self.textLine = QLineEdit()
        self.button = QPushButton("done")
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.textLine)
        layout.addWidget(self.button)
        # Set dialog layout
        self.setLayout(layout)

        global username
        user_type = self.textLine.text()
        print("username")

        self.connect(self.button, QtCore.SIGNAL("clicked()"), qApp, QtCore.SLOT("quit()"))


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





