from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os, sys
import numpy as np


file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(root_path)

import network.module_functions as function_compiler
from utils import checkpoints

class CamHistoryThread(QThread):
    cam_data = pyqtSignal(object)
    onprogress = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()
        self.cam_history_data = np.load(os.path.join(root_path, "cam_history", "cam_history_data.npz"))
        self.selectedIdx = self.cam_history_data['index']
        self.updating = False

    def __del__(self):
        self.wait()       

    def run(self):
        if self.isRunning(): self.quit()

        total = len(self.cam_history_data['data'])
        for idx, data in enumerate(self.cam_history_data['data']):
            
            progress = int((idx/total) * 100.0)

            if not self.updating:
                self.cam_data.emit(data)
                self.onprogress.emit(progress)

            self.msleep(100)

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
            self.msleep(150)        
        

class NetworkInitializationThread(QThread):
    onprogress = pyqtSignal(int)
    onmessage = pyqtSignal(str)
    oncompiled = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init()

    def run(self):

        #Import Network Module
        try:
            import network.VRN_64_dnn as config_module
        except Exception as e:
            self.onmessage.emit("No DNN Support. import gpuarray Support,, DNN support will be deprecated soon.")
            import network.VRN_64_gpuarray as config_module


        
        self.onmessage.emit("Load config and model files..")
        cfg = config_module.cfg
        model = config_module.get_model()


        #Compile Functions
        self.onmessage.emit('Compiling Theano Functions..')


        predict_function, colormap_function = function_compiler.make_functions(cfg, model)
        #Load Weights
        metadata, param_dict = checkpoints.load_weights(weightPath, model['l_out'])

