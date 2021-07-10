# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from QADemoGui import Ui_QADemoGui

style_files = {
    'white': 'qss/white.qss',
    'black': 'qss/black.qss',
}

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_QADemoGui()
    ui.setupUi(mainWindow)
    # set qss style "黑色炫酷"
    theme = 'white'
    qssStyle = open(style_files[theme], encoding='utf-8').read()
    mainWindow.setStyleSheet(qssStyle)
    # main loop
    mainWindow.show()
    sys.exit(app.exec_())
