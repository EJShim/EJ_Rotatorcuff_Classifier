import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt

#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir, os.pardir))
sys.path.insert(0, root_path)
# sys.setrecursionlimit(2000)

#Import Network
import network.VRN_64_TF as config_module

#Training Data Path
TRAIN_DATA_PATH = os.path.join(root_path, "data", "TrainData.npz")
TEST_DATA_PATH =  os.path.join(root_path, "data", "TestData.npz")


#Load Features and Targets
data_load = np.load(TRAIN_DATA_PATH)
features = data_load['features']
features = np.reshape(features, (features.shape[0], features.shape[2], features.shape[3], features.shape[4], features.shape[1]))
targets = data_load['targets']

#Load Test Features and Targets
test_data_load = np.load(TEST_DATA_PATH)
test_features = test_data_load['features']
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[2], test_features.shape[3], test_features.shape[4], test_features.shape[1]))
test_targets = test_data_load['targets']

#Get Configuration
cfg = config_module.cfg
max_epochs = cfg['max_epochs']
batch_size = cfg['batch_size']

#Initialize Figure
figure = plt.figure(1)
loss_ax = figure.add_subplot(211)
loss_ax.set_ylabel("loss")
loss_ax.set_xlabel("iterations")
loss_data = []

acc_ax = figure.add_subplot(212)
acc_ax.set_ylabel("accuracy")
acc_ax.set_xlabel("epoch")
acc_data = []

#Get TF Functions
x, y, keep_prob = config_module.get_model()
y_true = tf.placeholder(tf.int32)
sess = tf.InteractiveSession()

#Predict Functions
pred_classes = tf.argmax(y, axis=1)
pred_probs = tf.nn.softmax(y)

#Training Functions
learning_rate = tf.placeholder(tf.float32)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_true))
regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
loss = loss_op + 0.001*regularizer
trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
optimize = trainer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

lr = 0.002
sess.run(init)
for i in range(max_epochs):
    if i == 12: lr=0.0002

    idx = 0
    while idx < len(features):
        input_feed = features[idx:idx+batch_size]
        label_feed = targets[idx:idx+batch_size]
        idx = idx+batch_size
        _,loss_out = sess.run([optimize, loss_op], feed_dict={x:input_feed, y_true:label_feed, learning_rate:lr, keep_prob:0.0})
        
        loss_data.append(loss_out)
        loss_ax.plot(loss_data, 'r-')
        plt.pause(1e-1)
    
    save_path = saver.save(sess, os.path.join(file_path, 'weights', 'epoch'+str(i)+'model.ckpt'))
    
    #Run Test, Plot to the 
    score = []
    n_true = 0
    for i, t_target in enumerate(test_targets):

        pred, soft = sess.run([pred_classes, pred_probs], feed_dict={x:[test_features[i]], keep_prob:1.0})
        if pred[0] == t_target:
            n_true += 1
        score.append(soft[0][1])
    acc_data.append((n_true / 200) * 100.0)
    acc_ax.plot(acc_data, 'b-')
    plt.pause(1e-1)

    #Save ROC Data
    np.savez_compressed(os.path.join(file_path, os.pardir, 'roc_data' 'tf_epoch' + str(idx+1)), y=test_targets, score=score)

    #Shuffle Features  and Targets
    p = np.random.permutation(len(targets))
    features = features[p]
    targets = targets[p]

plt.show()