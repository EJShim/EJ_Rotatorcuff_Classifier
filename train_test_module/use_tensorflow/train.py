import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
plt.ion()

#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir, os.pardir))
sys.path.insert(0, root_path)
# sys.setrecursionlimit(2000)

#Import Network
import network.VRN_64_TF as config_module

#Training Data Path
TRAIN_DATA_PATH = os.path.join(root_path, "data", "TrainData.npz")


#Get Configuration
cfg = config_module.cfg
max_epochs = cfg['max_epochs']
batch_size = cfg['batch_size']

#Load Features and Targets
data_load = np.load(TRAIN_DATA_PATH)
features = data_load['features']
features = np.reshape(features, (features.shape[0], features.shape[2], features.shape[3], features.shape[4], features.shape[1]))
targets = data_load['targets']

#Get TF Functions

x, y, keep_prob = config_module.get_model()
y_true = tf.placeholder(tf.int32)


#Make Functions
pred_classes = tf.argmax(y, axis=1)
pred_probs = tf.nn.softmax(y)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_true))
trainer = tf.train.AdamOptimizer(learning_rate = 0.002)
optimize = trainer.minimize(loss_op)



#Saver
saver = tf.train.Saver()
data_loss = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(max_epochs):

        save_path = saver.save(sess, os.path.join(file_path, 'weights', 'epoch'+str(i)+'model.ckpt'))
        print("Epoch : ", i, "saved in %s"%save_path)
        idx = 0
        while idx < len(features):
            input_feed = features[idx:idx+batch_size]
            label_feed = targets[idx:idx+batch_size]
            idx = idx+batch_size

            loss, _ = sess.run([loss_op, optimize], feed_dict={x:input_feed, y_true:label_feed, keep_prob:0.0})

            data_loss.append(loss)
            if len(data_loss) > 30:                
                data_loss.pop(0)
                plt.clf()
            plt.plot(np.arange(len(data_loss)), data_loss, 'r-')
            plt.pause(0.0001)