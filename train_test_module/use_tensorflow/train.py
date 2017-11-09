import tensorflow as tf
import numpy as np
import sys, os


#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir, os.pardir))
sys.path.insert(0, root_path)
# sys.setrecursionlimit(2000)

#Import Network
import network.VRN_64_gpuarray as config_module

#Training Data Path
TRAIN_DATA_PATH = os.path.join(root_path, "data", "TrainData.npz")


#Get Configuration
cfg = config_module.cfg


#Get TF Functions
x, _, y = config_module.get_model()

#Get Ground Truth
y_true = tf.placeholder(tf.int32)


#Make Functions
pred_classes = tf.argmax(y, axis=1)
pred_probs = tf.nn.softmax(y)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_true))
trainer = tf.train.AdamOptimizer(learning_rate = 0.005)
optimize = trainer.minimize(loss_op)



#Saver
saver = tf.train.Saver()