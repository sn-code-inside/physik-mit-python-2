# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'wuerfel_gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(287, 240)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button_wuerfeln = QtWidgets.QPushButton(self.centralwidget)
        self.button_wuerfeln.setObjectName("button_wuerfeln")
        self.verticalLayout.addWidget(self.button_wuerfeln)
        self.label_anzeige = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_anzeige.setFont(font)
        self.label_anzeige.setText("")
        self.label_anzeige.setAlignment(QtCore.Qt.AlignCenter)
        self.label_anzeige.setObjectName("label_anzeige")
        self.verticalLayout.addWidget(self.label_anzeige)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Spielwürfel"))
        self.button_wuerfeln.setText(_translate("MainWindow", "Würfeln!"))

