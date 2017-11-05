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


true_volume_list = np.load(modelPath)['features']
true_image_list = []
for volume in true_volume_list:
    vol = volume[0]
    true_image_list.append(vol[32])
    true_image_list.append(np.rot90(vol, axes=(0,1))[32])
    true_image_list.append(np.rot90(vol, axes=(0,2))[32])


#Get Features and Targets
true_image_list = np.array(true_image_list)
true_targets = np.ones(len(true_image_list))

false_image_list = np.array(false_data_list)
false_targets = np.zeros(len(false_image_list))



#Concatenate True and False Data
features = np.concatenate((true_image_list, false_image_list))
targets = np.concatenate((true_targets, false_targets))


#Shuffle Features and targets
random_seed = np.arange(len(features))
np.random.shuffle(random_seed)

features = features[random_seed]
targets = targets[random_seed]

print(features.shape)
print(targets.shape)

np.savez_compressed(file_path + "/test_data", features=features, targets=targets)
