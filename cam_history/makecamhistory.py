TESTDATA_IDX = 39
RESULT = []


import os
import sys
import glob
import imp
import numpy as np
import scipy.ndimage
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
modelPath = os.path.join(root_path, "data", "TestData.npz")
sys.path.append(root_path)

#Import Local Modules
from utils import checkpoints
import network.VRN_64_gpuarray as config_module
import network.module_functions as function_compiler




#Initialize Network Graph
print("Import Pre-trained Network..")
print("Import Completed.")
print("Load config and model files..")
cfg = config_module.cfg
model = config_module.get_model()
#Compile Functions
print('Compiling Theano Functions..')
predict_function, colormap_function = function_compiler.make_functions(cfg, model)
print("Import Completed")


weight_path = os.path.join(root_path, "train_test_module", "tmp2")
weight_list = list(glob.iglob(str(weight_path) + '/*.npz', recursive = False))
weight_list.sort()

#Input Data = TestDAta[4]
inputData = np.asarray(np.load(modelPath)['features'], dtype=np.float32)[TESTDATA_IDX]


for idx, weight_file in enumerate(weight_list):
    #Load Weights
    print("Load ", weight_file)
    metadata, param_dict = checkpoints.load_weights(weight_file, model['l_out'])

    resolution = 64
    inputData = np.asarray(inputData.reshape(1, 1, resolution, resolution, resolution), dtype=np.float32)

    
    colorMap = colormap_function(inputData)

    #Compute Class Activation Map of Tear
    predIdx = 1
    fc1_weight = param_dict['fc.W']
    predWeights = fc1_weight[:,predIdx:predIdx+1]
    camsum = np.zeros((colorMap.shape[2], colorMap.shape[3], colorMap.shape[4]))
    for i in range(colorMap.shape[1]):
        camsum = camsum + predWeights[i] * colorMap[0,i,:,:,:]            
    camsum = scipy.ndimage.zoom(camsum, 16)

    #Normalize To 0-255
    tmp = camsum - np.amin(camsum)
    camsum = tmp / np.amax(tmp)               
    camsum *= 255.0
    camsum = camsum.astype(int)

    RESULT.append(camsum)


    print(idx, "-->", camsum.shape)


np.savez_compressed(str(os.path.join(file_path, "cam_history_data.npz")), index=TESTDATA_IDX, data=RESULT)