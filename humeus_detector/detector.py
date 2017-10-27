import os, sys
import glob
import numpy as np
import vtk
import scipy.ndimage
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))

modelPath = os.path.join(root_path, "data", "TestData.npz")
modelPath2 = os.path.join(root_path, "data", "TrainData.npz")
sys.path.append(root_path)


test = np.load(modelPath)['features']


image_slice = []
for volume in test:
    vol = volume[0]
    image_slice.append(vol[32])
    image_slice.append(np.rot90(vol, axes=(0,1))[32])
    image_slice.append(np.rot90(vol, axes=(0,2))[32])



image_slice = np.array(image_slice)
shape = image_slice.shape
image_slice = np.reshape(image_slice, (shape[0], 1, shape[1], shape[2]))

print(image_slice.shape)

for image in image_slice:
    plt.imshow(image[0], cmap='gray')
    plt.show()
    