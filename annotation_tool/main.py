# -*- coding: utf-8 -*-
import sys, os
from PyQt5.QtWidgets import *
from windows import AnnotationWindow





app = QApplication([])

window = AnnotationWindow()

window.showMaximized()
window.show()
sys.exit(app.exec_())
