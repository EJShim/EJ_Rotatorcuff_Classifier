import os,sys

import numpy as np
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))


data_load = np.load(os.path.join(file_path, "train_record_4block.npz"))

accuracy = data_load['accuracy']
# print(len(loss))
print(list(accuracy))
print("Maximum Accuracy : ", np.argmax(accuracy), max(accuracy))
#Plot Accuracy
figure = plt.figure(1)
ax = figure.add_subplot(111)
ax.set_title("Accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy (%)")
ax.grid(True)
ax.set_ylim([50, 100])
ax.set_xlim([0, 50])
ax.plot(accuracy, 'ro-', markersize=3)



#Plot Loss
loss = data_load['loss']
figure = plt.figure(2)
ax = figure.add_subplot(111)
ax.set_title("Loss")
ax.set_xlabel("minibatch")
ax.set_ylabel("loss")
ax.plot(loss)


epoch_length = int(len(loss)/49)
avg_loss = []
for i in range(0, len(loss), epoch_length):
    cur_epoch = loss[i:i+epoch_length]
    avg_loss.append(np.mean(cur_epoch))
print(avg_loss)
figure = plt.figure(3)
ax = figure.add_subplot(111)
ax.set_title("Average Loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.plot(avg_loss, 'r-')
ax.grid(True)



plt.show()
