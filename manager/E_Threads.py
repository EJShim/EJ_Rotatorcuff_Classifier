from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os, sys
import numpy as np
import scipy.ndimage

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(root_path)


class CamHistoryThread(QThread):
    cam_data = pyqtSignal(object)
    onprogress = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()

        data_load = np.load(os.path.join(root_path, "train_test_module",  "train_record_3block.npz"))
        cam_data = data_load['cam']
        deconv_rate =  64 /cam_data.shape[1] 

        self.cam_history_data = []
        for data in cam_data:
            data = scipy.ndimage.zoom(data, deconv_rate)
            data = data / 8
            data *= 255.0
            data = data.astype(int)
            self.cam_history_data.append(data)
        

        # self.cam_history_data = np.load(os.path.join(root_path, "cam_history", "cam_history_data.npz"))
        self.selectedIdx = 116
        self.updating = False

    def __del__(self):
        self.wait()       

    def run(self):
        if self.isRunning(): self.quit()

        total = len(self.cam_history_data)
        for idx, data in enumerate(self.cam_history_data):
            
            progress = int((idx/total) * 100.0)

            if not self.updating:
                self.cam_data.emit(data)
                self.onprogress.emit(progress)

            self.msleep(20/1000)

        self.quit()

#Multi-Thread Codes
class ListAnimationThread(QThread):
    predRandom = pyqtSignal(bool)


    def __init__(self, parent=None):
        super().__init__()        
        

    # def __del__(self):
    #     self.wait()    

    def run(self):
        while self.isRunning():
            self.predRandom.emit(True)            
            self.msleep(1000)