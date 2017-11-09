import tensorflow as tf


keep_prob = tf.placeholder(dtype=tf.float32)

oneone = 1.0 - (1.0 - keep_prob)*0.95


with tf.Session() as sess:
    ha = sess.run(oneone, feed_dict={keep_prob:0.0})

    print(ha)