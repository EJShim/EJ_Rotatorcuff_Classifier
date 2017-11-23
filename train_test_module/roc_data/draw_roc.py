import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from sklearn.metrics import auc, roc_curve

file_path = os.path.dirname(os.path.abspath(__file__))


file_list = list(glob.iglob(file_path + '/2block/*.npz'))
file_list.sort()

auc_data = []
for data_path in file_list:
    data = np.load(data_path)


    # fpr, tpr = get_roc(data['y'], data['score'])
    fpr, tpr, th = roc_curve(data['y'], data['score'])


    auc_value = auc(fpr, tpr)
    data_name = os.path.basename(data_path)
    auc_data.append(auc_value)


    plt.plot(fpr, tpr, '-', label='%s  (AUC = %0.3f)'%(data_name[:-4], auc_value))


print(fpr.tolist())
print(tpr.tolist())




figure = plt.figure(2)
ax = figure.add_subplot(111)
ax.set_xlabel("epoch")
ax.set_xlim([0, 50])
ax.set_ylabel("AUC")
ax.grid(True)
ax.plot(auc_data, 'bo-', markersize=3)
ax.legend(loc='best')


print(auc_data)

print(np.argmax(auc_data[13:]), max(auc_data[13:]))



plt.show()