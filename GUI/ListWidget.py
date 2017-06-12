from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class E_ListWidget(QWidget):
    def __init__(self, parent = None):
        super(E_ListWidget, self).__init__(parent)

        self.m_cWidget = QWidget()
        self.setCentralWidget(self.m_cWidget)    
