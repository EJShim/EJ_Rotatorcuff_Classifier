import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import network as ejnet



tr_in, tr_out = ejnet.get_traditional_model()

file_path = os.path.dirname(os.path.realpath(__file__))
data_load = np.load(file_path + "/train_data.npz")

features = data_load['features']
targets = data_load['targets']

print(features.shape)
print(targets.shape)



#Test Forward Propagation
with tf.Session() as sess:
    trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
    sess.run(tf.global_variables_initializer())


    batch_size = 16
    max_epochs = 250

    
    for i in range(0, max_epochs):
        print("epoch : ", i)
        idx = 0
        while idx < len(features):
            input_feed = features[idx:idx + batch_size]
            labels = targets[idx:idx+batch_size]
            idx = idx+batch_size

            #Reshape input_feed
            input_feed = np.reshape(input_feed, (input_feed.shape[0], input_feed.shape[1], input_feed.shape[2], 1))

            pred_classes = tf.argmax(tr_out, axis=1)
            pred_probs = tf.nn.softmax(tr_out)


            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tr_out, 
                labels=tf.cast(labels, dtype=tf.int32)))

            
            optimize = trainer.minimize(loss_op)

            loss_out, _ = sess.run([loss_op, optimize], feed_dict={tr_in:input_feed})


            out.append(loss_out)
            ax.plot(np.arange(len(out)), out, 'r-')
            fig.canvas.draw()
            
            print(loss_out)
        plt.show(block=True)
            

