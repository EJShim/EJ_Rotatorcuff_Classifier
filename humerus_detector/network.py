import tensorflow as tf
import numpy as np


def get_traditional_model():
    l_input = tf.placeholder(tf.float32, [None, 64, 64, 1])
    l_conv1 = tf.layers.conv2d(inputs=l_input, filters=96, kernel_size=(10,10), strides=(1, 1), name='l_conv1')
    l_pool1 = tf.layers.max_pooling2d(inputs = l_conv1, pool_size=(3,3), strides=(2,2), name="l_pool1")
    l_bn1 = tf.layers.batch_normalization(l_pool1, name="l_bn1")
    l_conv2 = tf.layers.conv2d(inputs=l_bn1, filters = 256, kernel_size=(5,5), strides=(1,1), name='l_conv2')
    l_pool2 = tf.layers.max_pooling2d(inputs=l_conv2, pool_size=(3,3), strides=(2,2), name='l_pool2')
    l_bn2 = tf.layers.batch_normalization(l_pool2, name='l_bn2')
    l_conv3 = tf.layers.conv2d(inputs=l_bn2, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv3')
    l_conv4 = tf.layers.conv2d(inputs=l_conv3, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv4')
    l_conv5 = tf.layers.conv2d(inputs=l_conv4, filters=256, kernel_size=(3,3), strides=(1, 1), name='l_conv5')
    l_pool3 = tf.layers.max_pooling2d(inputs=l_conv5, pool_size=(4,4), strides=(2,2), name='l_pool3')

    #Flatten Here..
    l_fc0 = tf.contrib.layers.flatten(inputs=l_pool3)    
    l_fc1 = tf.layers.dense(inputs=l_fc0, units=4096, name='l_fc1')
    l_fc2 = tf.layers.dense(inputs=l_fc1, units=2, name='l_fc2')

    return l_input, l_fc2

def get_fcn_model():
    l_input = tf.placeholder(tf.float32, [None, None, None, 1])
    l_conv1 = tf.layers.conv2d(inputs=l_input, filters=96, kernel_size=(10,10), strides=(1, 1), name='l_conv1')
    l_pool1 = tf.layers.max_pooling2d(inputs = l_conv1, pool_size=(3,3), strides=(2,2), name="l_pool1")
    l_bn1 = tf.layers.batch_normalization(l_pool1, name="l_bn1")
    l_conv2 = tf.layers.conv2d(inputs=l_bn1, filters = 256, kernel_size=(5,5), strides=(1,1), name='l_conv2')
    l_pool2 = tf.layers.max_pooling2d(inputs=l_conv2, pool_size=(3,3), strides=(2,2), name='l_pool2')
    l_bn2 = tf.layers.batch_normalization(l_pool2, name='l_bn2')
    l_conv3 = tf.layers.conv2d(inputs=l_bn2, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv3')
    l_conv4 = tf.layers.conv2d(inputs=l_conv3, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv4')
    l_conv5 = tf.layers.conv2d(inputs=l_conv4, filters=256, kernel_size=(3,3), strides=(1, 1), name='l_conv5')
    l_pool3 = tf.layers.max_pooling2d(inputs=l_conv5, pool_size=(4,4), strides=(2,2), name='l_pool3')
    l_fc1 = tf.layers.conv2d(inputs=l_pool3, filters=4096, kernel_size=(1,1), name='l_fc1')
    l_fc2 = tf.layers.conv2d(inputs=l_fc1, filters=2, kernel_size=(1,1), name='l_fc2')

    print(l_fc2.shape)
    return l_input, l_fc2



