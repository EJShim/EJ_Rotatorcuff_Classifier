import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt

#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.insert(0, root_path)
# sys.setrecursionlimit(2000)

#Import Network
import network.VRN_64_TF as config_module

#Training Data Path
TRAIN_DATA_PATH = os.path.join(root_path, "data", "TrainData.npz")
TEST_DATA_PATH =  os.path.join(root_path, "data", "TestData.npz")

def subsample_array(arr):
    ratio = int(len(arr)/100)
    if ratio == 0:
        return arr

    return arr[0::ratio]


#Load Features and Targets
data_load = np.load(TRAIN_DATA_PATH)
features = data_load['features']
features = np.reshape(features, (features.shape[0], features.shape[2], features.shape[3], features.shape[4], features.shape[1]))
targets = data_load['targets']

#Load Test Features and Targets
test_data_load = np.load(TEST_DATA_PATH)
test_features = test_data_load['features']
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[2], test_features.shape[3], test_features.shape[4], test_features.shape[1]))
cam_features = test_features[116]
cam_features = np.reshape(cam_features, (1, cam_features.shape[0], cam_features.shape[1], cam_features.shape[2], cam_features.shape[3]))
test_targets = test_data_load['targets']


#Get Configuration
cfg = config_module.cfg
max_epochs = cfg['max_epochs']
batch_size = cfg['batch_size']

#Initialize Figure
figure = plt.figure(1)
loss_ax = figure.add_subplot(211)
loss_data = []

acc_ax = figure.add_subplot(212)
acc_data = []

#Get TF Functions
x, y, keep_prob, last_conv = config_module.get_very_shallow_model()
y_true = tf.placeholder(tf.int32)
sess = tf.InteractiveSession()

#Predict Functions
y_deter = tf.layers.flatten(y)
pred_classes = tf.argmax(y_deter, axis=1)
pred_probs = tf.nn.softmax(y_deter)
gr = tf.get_default_graph()
last_weight = gr.get_tensor_by_name('fc/kernel:0')
last_conv = last_conv[0]
last_weight = last_weight[:,:,:,:,1]
class_activation_map =tf.nn.relu( tf.reduce_sum(tf.multiply(last_weight, last_conv), axis=3))
cam_data = []

#Training Functions
learning_rate = tf.placeholder(tf.float32)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_true))
regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
loss = loss_op + 0.001*regularizer
trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

#batch normalizer?
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimize = trainer.minimize(loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

lr = 0.002
sess.run(init)
for epoch in range(max_epochs):
    #Run Test, Get Accuracy 
    score = []
    n_true = 0
    #Class Activation Map Every Batch!
    cam = sess.run(class_activation_map, feed_dict={x:cam_features, keep_prob:1.0})
    cam_data.append(cam)
    for idx, t_target in enumerate(test_targets):
        pred, soft= sess.run([pred_classes, pred_probs], feed_dict={x:[test_features[idx]], keep_prob:1.0})
        if pred[0] == t_target:
            n_true += 1
        score.append(soft[0][1])
    
    acc_data.append((n_true / 200) * 100.0)
    acc_ax.cla()
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_xlabel("epoch")
    acc_ax.plot(acc_data, 'b-')

    loss_ax.cla()
    loss_ax.set_ylabel("loss")
    loss_ax.set_xlabel("iterations")
    loss_ax.plot(loss_data, 'r-')
    plt.pause(1e-45)
    #Save ROC Data
    np.savez_compressed(os.path.join(file_path,  'roc_data', 'tf_epoch' + str(epoch)), y=test_targets, score=score)
    np.savez_compressed(os.path.join(file_path, 'train_record'), loss=loss_data, accuracy=acc_data, cam=cam_data)

    #Training Module
    if epoch == 12: lr=0.0002
    idx = 0
    while idx < len(features):
        input_feed = features[idx:idx+batch_size]
        label_feed = targets[idx:idx+batch_size]
        idx = idx+batch_size


        _,loss_out = sess.run([optimize, loss_op], feed_dict={x:input_feed, y_true:label_feed, learning_rate:lr, keep_prob:0.0})
    
        
        loss_data.append(loss_out)
        if epoch < 2:     
            print(loss_out)
            loss_ax.cla()
            loss_ax.set_ylabel("loss")
            loss_ax.set_xlabel("iterations")
            loss_ax.plot(loss_data, 'r-')
            plt.pause(1e-45)
    save_path = saver.save(sess, os.path.join(file_path, 'weights', 'epoch'+str(epoch)+'model.ckpt'))

    #Shuffle Features  and Targets
    p = np.random.permutation(len(targets))
    features = features[p]
    targets = targets[p]

plt.show()