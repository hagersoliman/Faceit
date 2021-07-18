# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import Qt

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 500)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.image_label3 = QtWidgets.QLabel(Form)
        self.image_label3.setObjectName("image_label3")
        self.verticalLayout.addWidget(self.image_label3,1)
        self.image_label3.setAlignment(Qt.AlignCenter)
        self.image_label3.setStyleSheet("background-color: lightgreen")

        self.horizontalLayout2 = QtWidgets.QHBoxLayout()

        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setObjectName("image_label")
        self.horizontalLayout2.addWidget(self.image_label)

        self.image_label2 = QtWidgets.QLabel(Form)
        self.image_label2.setObjectName("image_label2")
        self.horizontalLayout2.addWidget(self.image_label2)

        self.verticalLayout.addLayout(self.horizontalLayout2,6)

        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt,1)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        self.image_label.setText(_translate("Form", "Receive"))
        self.image_label2.setText(_translate("Form", "Send"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label3.setText(_translate("Form", "logo"))
        self.control_bt.setText(_translate("Form", "Start"))
