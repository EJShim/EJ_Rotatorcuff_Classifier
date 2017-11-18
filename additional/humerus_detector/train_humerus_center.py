import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import network as ejnet



#Initialize Figure Options
xData = []
yData = []

plt.ion()

file_path = os.path.dirname(os.path.realpath(__file__))
data_load = np.load(file_path + "/train_data.npz")

features = data_load['features']
targets = data_load['targets']

print(features.shape)
print(targets.shape)

#Config
batch_size = 32
max_epochs = 3


#Make TF Functions
tr_label = tf.placeholder(tf.int32)
tr_in, tr_out = ejnet.get_fcn_model()
pred_classes = tf.argmax(tr_out, axis=1)
pred_probs = tf.nn.softmax(tr_out)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tr_out, labels=tr_label))
trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
optimize = trainer.minimize(loss_op)

saver = tf.train.Saver()



#Test Forward Propagation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #Run Training
    for i in range(0, max_epochs):
        print("epoch : ", i)
        idx = 0
        while idx < len(features):
            input_feed = features[idx:idx + batch_size]
            labels = targets[idx:idx+batch_size]
            idx = idx+batch_size

            #Reshape input_feed
            input_feed = np.reshape(input_feed, (input_feed.shape[0], input_feed.shape[1], input_feed.shape[2], 1))            
            loss_out, _ = sess.run([loss_op, optimize], feed_dict={tr_in:input_feed, tr_label :labels})            
            
            print(loss_out)



            # xData.append(idx)
            yData.append(loss_out)
            plt.plot(np.arange(len(yData)), yData, 'r-', linewidth=1.5, markersize=4)
            plt.pause(0.0001)

    
    # Save the variables to disk.
    save_path = saver.save(sess, file_path + "/weights/model.ckpt")
    print("Model saved in file: %s" % save_path)
            

