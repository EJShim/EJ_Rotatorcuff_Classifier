import tensorflow as tf
import numpy as np


def get_traditional_model():
    l_input = tf.placeholder(tf.float32, [None, None, None, 1])
    l_conv1 = tf.layers.conv2d(inputs=l_input, filters=96, kernel_size=(10,10), strides=(1, 1), name='l_conv1')
    l_pool1 = tf.layers.average_pooling2d(inputs = l_conv1, pool_size=(3,3), strides=(2,2), name="l_pool1")
    l_bn1 = tf.layers.batch_normalization(l_pool1, name="l_bn1")
    l_conv2 = tf.layers.conv2d(inputs=l_bn1, filters = 256, kernel_size=(5,5), strides=(1,1), name='l_conv2')
    l_pool2 = tf.layers.average_pooling2d(inputs=l_conv2, pool_size=(3,3), strides=(2,2), name='l_pool2')
    l_bn2 = tf.layers.batch_normalization(l_pool2, name='l_bn2')
    l_conv3 = tf.layers.conv2d(inputs=l_bn2, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv3')
    l_conv4 = tf.layers.conv2d(inputs=l_conv3, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv4')
    l_conv5 = tf.layers.conv2d(inputs=l_conv4, filters=256, kernel_size=(3,3), strides=(1, 1), name='l_conv5')
    l_pool3 = tf.layers.average_pooling2d(inputs=l_conv5, pool_size=(4,4), strides=(2,2), name='l_pool3')

    #Flatten Here..
    # l_fc0 = tf.contrib.layers.flatten(inputs=l_pool3)    
    l_fc1 = tf.layers.dense(inputs=l_pool3, units=4096, name='l_fc1')
    l_fc2 = tf.layers.dense(inputs=l_fc1, units=2, name='l_fc2')

    print(l_fc2.shape)

    return l_input, l_fc2

def get_fcn_model():
    l_input = tf.placeholder(tf.float32, [None, None, None, 1])
    l_conv1 = tf.layers.conv2d(inputs=l_input, filters=96, kernel_size=(10,10), strides=(1, 1), name='l_conv1')
    l_pool1 = tf.layers.average_pooling2d(inputs = l_conv1, pool_size=(3,3), strides=(2,2), name="l_pool1")
    l_bn1 = tf.layers.batch_normalization(l_pool1, name="l_bn1")
    l_conv2 = tf.layers.conv2d(inputs=l_bn1, filters = 256, kernel_size=(5,5), strides=(1,1), name='l_conv2')
    l_pool2 = tf.layers.average_pooling2d(inputs=l_conv2, pool_size=(3,3), strides=(2,2), name='l_pool2')
    l_bn2 = tf.layers.batch_normalization(l_pool2, name='l_bn2')
    l_conv3 = tf.layers.conv2d(inputs=l_bn2, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv3')
    l_conv4 = tf.layers.conv2d(inputs=l_conv3, filters=384, kernel_size=(3,3), strides=(1, 1), name='l_conv4')
    l_conv5 = tf.layers.conv2d(inputs=l_conv4, filters=256, kernel_size=(3,3), strides=(1, 1), name='l_conv5')
    l_pool3 = tf.layers.average_pooling2d(inputs=l_conv5, pool_size=(4,4), strides=(2,2), name='l_pool3')
    l_fc1 = tf.layers.conv2d(inputs=l_pool3, filters=4096, kernel_size=(1,1), name='l_fc1')
    l_fc2 = tf.layers.conv2d(inputs=l_fc1, filters=2, kernel_size=(1,1), name='l_fc2')

    print(l_fc2.shape)
    return l_input, l_fc2

def get_egg_model():

    #Deeper network like vgg
    l_input = tf.placeholder(tf.float32, [None, None, None, 1])
    l_conv1 = tf.layers.conv2d(inputs=l_input, filters=4, kernel_size=(3,3), name='l_conv1')
    l_pool1 = tf.layers.max_pooling2d(inputs = l_conv1, pool_size=(2,2), strides=(1,1), name="l_pool1")
    l_conv2 = tf.layers.conv2d(inputs=l_pool1, filters=4, kernel_size=(3,3), name='l_conv2')
    l_pool2 = tf.layers.max_pooling2d(inputs = l_conv2, pool_size=(2,2), strides=(1,1), name="l_pool2")
    l_conv3 = tf.layers.conv2d(inputs=l_pool2, filters=4, kernel_size=(3,3), name='l_conv3')
    l_pool3 = tf.layers.max_pooling2d(inputs = l_conv3, pool_size=(2,2), strides=(1,1), name="l_pool3")
    l_conv4 = tf.layers.conv2d(inputs=l_pool3, filters=8, kernel_size=(3,3), name='l_conv4')
    l_pool4 = tf.layers.max_pooling2d(inputs = l_conv4, pool_size=(2,2), strides=(1,1), name="l_pool4")
    l_conv5 = tf.layers.conv2d(inputs=l_pool4, filters=8, kernel_size=(3,3), name='l_conv5')
    l_pool5 = tf.layers.max_pooling2d(inputs = l_conv5, pool_size=(2,2), strides=(1,1), name="l_pool5")
    l_conv6 = tf.layers.conv2d(inputs=l_pool5, filters=8, kernel_size=(3,3), name='l_conv6')
    l_pool6 = tf.layers.max_pooling2d(inputs = l_conv6, pool_size=(2,2), strides=(1,1), name="l_pool6")
    l_conv7 = tf.layers.conv2d(inputs=l_pool6, filters=16, kernel_size=(3,3), name='l_conv7')
    l_pool7 = tf.layers.max_pooling2d(inputs = l_conv7, pool_size=(2,2), strides=(1,1), name="l_pool7")
    l_conv8 = tf.layers.conv2d(inputs=l_pool7, filters=16, kernel_size=(3,3), name='l_conv8')
    l_pool8 = tf.layers.max_pooling2d(inputs = l_conv8, pool_size=(2,2), strides=(1,1), name="l_pool8")
    l_conv9 = tf.layers.conv2d(inputs=l_pool8, filters=16, kernel_size=(3,3), name='l_conv9')
    l_pool9 = tf.layers.max_pooling2d(inputs = l_conv9, pool_size=(2,2), strides=(1,1), name="l_poo9")
    l_conv10 = tf.layers.conv2d(inputs=l_pool9, filters=32, kernel_size=(3,3), name='l_conv10')
    l_pool10 = tf.layers.max_pooling2d(inputs = l_conv10, pool_size=(2,2), strides=(1,1), name="l_pool10")
    l_conv11 = tf.layers.conv2d(inputs=l_pool10, filters=32, kernel_size=(3,3), name='l_conv11')
    l_pool11 = tf.layers.max_pooling2d(inputs = l_conv11, pool_size=(2,2), strides=(1,1), name="l_pool11")
    l_conv12 = tf.layers.conv2d(inputs=l_pool11, filters=32, kernel_size=(3,3), name='l_conv12')
    l_pool12 = tf.layers.max_pooling2d(inputs = l_conv12, pool_size=(2,2), strides=(1,1), name="l_pool12")
    l_conv13 = tf.layers.conv2d(inputs=l_pool12, filters=64, kernel_size=(3,3), name='l_conv13')
    l_pool13 = tf.layers.max_pooling2d(inputs = l_conv13, pool_size=(2,2), strides=(1,1), name="l_pool13")
    l_conv14 = tf.layers.conv2d(inputs=l_pool13, filters=64, kernel_size=(3,3), name='l_conv14')
    l_pool14 = tf.layers.max_pooling2d(inputs = l_conv14, pool_size=(2,2), strides=(1,1), name="l_pool14")
    l_conv15 = tf.layers.conv2d(inputs=l_pool14, filters=64, kernel_size=(3,3), name='l_conv15')
    l_pool15 = tf.layers.max_pooling2d(inputs = l_conv15, pool_size=(2,2), strides=(1,1), name="l_pool15")
    l_conv16 = tf.layers.conv2d(inputs=l_pool15, filters=256, kernel_size=(3,3), name='l_conv16')
    l_pool16 = tf.layers.max_pooling2d(inputs = l_conv16, pool_size=(2,2), strides=(1,1), name="l_pool16")
    l_conv17 = tf.layers.conv2d(inputs=l_pool16, filters=512, kernel_size=(4,4), name='l_conv17')
    l_pool17 = tf.layers.max_pooling2d(inputs = l_conv17, pool_size=(3,3), strides=(1,1), name="l_pool17")
    l_conv18 = tf.layers.conv2d(inputs=l_pool17, filters=1024, kernel_size=(4,4), name='l_conv18')
    l_pool18 = tf.layers.max_pooling2d(inputs = l_conv18, pool_size=(3,3), strides=(1,1), name="l_pool18")
    l_conv19 = tf.layers.conv2d(inputs=l_pool18, filters=2048, kernel_size=(4,4), name='l_conv19')
    l_pool19 = tf.layers.max_pooling2d(inputs = l_conv19, pool_size=(3,3), strides=(1,1), name="l_pool19")
    l_fc1 = tf.layers.conv2d(inputs=l_pool19, filters=4096, kernel_size=(1,1), name='l_fc1')
    l_fc2 = tf.layers.conv2d(inputs=l_fc1, filters=2, kernel_size=(1,1), name='l_fc2')

    return l_input, l_fc2

