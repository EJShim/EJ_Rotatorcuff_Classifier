import matplotlib.pyplot as plt
import numpy as np
import glob
import os

file_path = os.path.dirname(os.path.abspath(__file__))



for data_path in glob.iglob(file_path + "/*.npz"):
    data = np.load(data_path)

    plt.plot(data['x'], data['y'], 'ro-')

plt.show()