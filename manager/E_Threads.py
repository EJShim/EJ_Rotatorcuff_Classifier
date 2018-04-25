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
    cam_data = pyqtSignal(object, int)    

    def __init__(self, parent=None):
        super().__init__()

        data_load = np.load(os.path.join(root_path, "train_test_module",  "train_record_2block.npz"))
        cam_data = data_load['cam']
        deconv_rate =  64 /cam_data.shape[1]
        self.cam_history_data = []
        for data in cam_data:
            data = scipy.ndimage.zoom(data, deconv_rate)
            data = data / 8
            data *= 255.0
            data = data.astype(int)
            self.cam_history_data.append(data)
            
        self.selectedIdx = 116

        self.total_memory = len(self.cam_history_data)
        self.current_idx = 0

        #SIGNAL RECEIVE
        parent.update_cam.connect(self.on_signal)

    def __del__(self):
        self.wait()       

    def run(self):
        self.current_idx = 0
        #Emit First Siganl
        self.cam_data.emit(self.cam_history_data[0], 0)
        self.quit()
    
    def on_signal(self):
        self.current_idx += 1
        if self.current_idx >= self.total_memory:             
            return        
        progress = int((self.current_idx/(self.total_memory-1)) * 100.0)
        self.cam_data.emit(self.cam_history_data[self.current_idx], progress)        
        

    
    

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