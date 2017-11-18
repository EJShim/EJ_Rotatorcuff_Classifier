# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from gui.MainFrm import E_MainWindow


app = QApplication([])

window = E_MainWindow()

window.showMaximized()
window.show()
sys.exit(app.exec_())
