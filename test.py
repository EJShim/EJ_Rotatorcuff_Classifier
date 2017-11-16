import numpy as np


data_load = np.load("./train_test_module/use_tensorflow/train_record.npz")
accuracy = data_load['accuracy']
print(accuracy)
print(len(accuracy))