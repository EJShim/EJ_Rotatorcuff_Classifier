# -*- coding: utf-8 -*-
import sys, os
from PyQt5.QtWidgets import *
from windows import AnnotationWindow
import Style




app = QApplication([])
app.setStyleSheet(Style.styleData)  


window = AnnotationWindow()

window.showMaximized()
window.show()
sys.exit(app.exec_())
