import numpy as np
import matplotlib.pyplot as plt

4block_auc = [0.50917431192660545, 0.90109890109890112, 0.9611856033874383, 0.97650972880330689, 0.97106563161609027, 0.92751285411835871, 0.92499243875390658, 0.94787781026313134, 0.95201129146083274, 0.96854521625163836, 0.96985583224115335, 0.9174311926605504, 0.96148805323117248, 0.96884766609537254, 0.96955338239741917, 0.97550156265752597, 0.97560237927210414, 0.97429176328258893, 0.97368686359512047, 0.97177134791813691, 0.97378768020969853, 0.97187216453271497, 0.9707631817723561, 0.97056154854319987, 0.97368686359512047, 0.97328359713680812, 0.97398931343885475, 0.97409013005343281, 0.97419094666801087, 0.97096481500151222, 0.97217461437644925, 0.97529992942836974, 0.97217461437644948, 0.97288033067849589, 0.97288033067849577, 0.97358604698054241, 0.97338441375138618,0.97146889807440251, 0.97459421312632322, 0.97328359713680812, 0.97398931343885464, 0.97378768020969853, 0.97469502974090128, 0.97237624760560548, 0.97318278052223017, 0.97187216453271497, 0.97207379776187119, 0.97237624760560537, 0.97096481500151233, 0.97338441375138618, 0.97177134791813691, 0.9722754309910272, 0.97318278052223006, 0.97136808145982456, 0.97116644823066844, 0.97288033067849577, 0.97388849682427669, 0.97076318177235599, 0.97348523036596435, 0.97267869744933966, 0.97146889807440262, 0.97217461437644925, 0.97136808145982456, 0.97328359713680812, 0.97237624760560548, 0.97167053130355885, 0.97177134791813691, 0.97519911281379179, 0.97328359713680812, 0.97025909869946569]

4block_acc = [45.5, 76.0, 83.0, 76.5, 75.0, 80.0, 90.0, 90.5, 85.0, 89.5, 92.0, 90.0, 93.5, 93.5, 95.0, 94.5, 95.5, 95.0, 95.0, 95.5, 95.0, 95.5, 95.0, 95.5, 95.5, 95.5, 95.5, 95.0, 95.5, 93.5, 95.5, 96.0, 96.0, 95.5, 96.0, 96.0, 96.0, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.5, 95.0, 95.5, 95.5, 94.5, 95.5, 95.5, 95.0, 95.5, 95.5, 95.0, 95.0, 95.0, 94.5, 95.5, 94.5, 95.0, 95.0, 95.5, 95.5, 95.0, 95.5, 95.0, 95.0, 95.0]
acc = [x*0.01 for x in acc]

plt.figure(1)
plt.plot(auc, 'r--', markersize=3, label="AUC")
plt.plot(acc, 'b-', markersize=3, label="Accuracy")
plt.xlabel("Epoch")
plt.legend(loc='best')





plt.show()

