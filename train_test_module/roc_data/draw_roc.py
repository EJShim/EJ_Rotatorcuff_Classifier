import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from sklearn.metrics import auc

file_path = os.path.dirname(os.path.abspath(__file__))



plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Rotator Cuff Classification')

plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0,1],[0,1], color='navy', linestyle='--', linewidth=0.5)


file_list = list(glob.iglob(file_path + '/*.npz'))
file_list.sort()


for data_path in file_list:
    data = np.load(data_path)

    data_name = os.path.basename(data_path)
    auc_value = auc(data['x'], data['y'])


    plt.plot(data['x'], data['y'], '-', label='%s  (AUC = %0.3f)'%(data_name[:-4], auc_value))
    
plt.legend(loc='best')
plt.show()