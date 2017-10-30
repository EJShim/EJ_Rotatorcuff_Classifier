import os, sys
import glob
import numpy as np
import vtk
import scipy.ndimage
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(root_path)




modelPath = os.path.join(root_path, "data", "TestData.npz")
false_path_list = list(glob.iglob(file_path + '/none_humerus_data/*.npz', recursive = False))



false_data_list = []
#Import False Data
for false_path in false_path_list:
    features = np.load(false_path)['features']
    for feature in features:
        false_data_list.append(feature)

false_data_list = np.array(false_data_list)


print("False Data : ", false_data_list.shape)

    



true_volume_list = np.load(modelPath)['features']
true_image_list = []
for volume in true_volume_list:
    vol = volume[0]
    true_image_list.append(vol[32])
    true_image_list.append(np.rot90(vol, axes=(0,1))[32])
    true_image_list.append(np.rot90(vol, axes=(0,2))[32])



true_image_list = np.array(true_image_list)
print(true_image_list.shape)
# shape = image_slice.shape
# image_slice = np.reshape(image_slice, (shape[0], 1, shape[1], shape[2]))

# print(image_slice.shape)

