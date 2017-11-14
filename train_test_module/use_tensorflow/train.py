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
# lr = cfg['lr']
# decay = (lr[0]-lr[1])/max_epochs



#Get TF Functions
x, y, keep_prob = config_module.get_model()
y_true = tf.placeholder(tf.int32)
sess = tf.InteractiveSession()


learning_rate = tf.placeholder(tf.float32)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_true))
regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
loss = loss_op + 0.001*regularizer



#Load Features and Targets
data_load = np.load(TRAIN_DATA_PATH)
features = data_load['features']
features = np.reshape(features, (features.shape[0], features.shape[2], features.shape[3], features.shape[4], features.shape[1]))
targets = data_load['targets']


trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
optimize = trainer.minimize(loss)

init = tf.global_variables_initializer()


#Saver
saver = tf.train.Saver()
data_loss = []
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

        print(loss_out)

        data_loss.append(loss_out)
        if len(data_loss) > 30:                
            data_loss.pop(0)
            plt.clf()

        
        plt.plot(np.arange(len(data_loss)), data_loss, 'ro-')
        plt.pause(0.0001)
    
    save_path = saver.save(sess, os.path.join(file_path, 'weights', 'epoch'+str(i)+'model.ckpt'))
    plt.title("Epoch "+str(i))
    print("Epoch : ", i, "saved in %s"%save_path)

    #Shuffle Features  and Targets
    p = np.random.permutation(len(targets))
    features = features[p]
    targets = targets[p]


plt.show()