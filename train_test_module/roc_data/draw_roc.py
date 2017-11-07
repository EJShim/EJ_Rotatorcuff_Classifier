import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from sklearn.metrics import auc, roc_curve

file_path = os.path.dirname(os.path.abspath(__file__))



plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Rotator Cuff Classification')

plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0,1],[0,1], color='navy', linestyle='--', linewidth=0.5)



def get_roc(y, score):#Deprecated
    roc_x = []
    roc_y = []

    thr = np.linspace(0, 1, 100)

    T = sum(y)
    F = len(y) - T

    FP=0
    TP=0

    for th in thr:
        for i in range(0, len(score)):
            if score[i]> th:
                if(y[i]==1):
                    TP += 1
                if(y[i]==0):
                    FP += 1

        sensitivity = FP/F
        specificity = TP/T


        roc_x.append(FP/F)
        roc_y.append(TP/T)
        FP=0
        TP=0

    return roc_x, roc_y



file_list = list(glob.iglob(file_path + '/*.npz'))
file_list.sort()

for data_path in file_list:
    data = np.load(data_path)


    # fpr, tpr = get_roc(data['y'], data['score'])
    fpr, tpr, th = roc_curve(data['y'], data['score'])


    auc_value = auc(fpr, tpr)
    data_name = os.path.basename(data_path)


    plt.plot(fpr, tpr, '-', label='%s  (AUC = %0.3f)'%(data_name[:-4], auc_value))
    
plt.legend(loc='best')
plt.show()