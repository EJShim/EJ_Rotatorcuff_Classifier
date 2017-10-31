import tensorflow as tf
import numpy as np
import os

import network as ejnet



tr_in, tr_out = ejnet.get_traditional_model()

file_path = os.path.dirname(os.path.realpath(__file__))
data_load = np.load(file_path + "/train_data.npz")

features = data_load['features']
targets = data_load['targets']

print(features.shape)
print(targets.shape)