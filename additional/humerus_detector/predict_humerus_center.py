import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import network as ejnet
import scipy.ndimage



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


#Initialize Figure Options

file_path = os.path.dirname(os.path.realpath(__file__))
data_load = np.load(file_path + "/train_data.npz")

features = data_load['features']
targets = data_load['targets']



#Make TF Functions
tr_in, tr_out = ejnet.get_fcn_model()
pred_classes = tf.argmax(tr_out, axis=3)
pred_probs = tf.nn.softmax(tr_out)

saver = tf.train.Saver()



#Test Forward Propagation
with tf.Session() as sess:
    saver.restore(sess, file_path + "/weights/model.ckpt")
    # sess.run(tf.global_variables_initializer())
    print("Model restored.")

    
    img = scipy.ndimage.imread(file_path + "/samples/4.jpg")
    img = rgb2gray(img)

    plt.figure()
    plt.imshow(img, cmap='gray')



    input_feed = img
    input_feed = np.reshape(input_feed, (1, input_feed.shape[0], input_feed.shape[1], 1))
    output = sess.run(pred_probs, feed_dict={tr_in:input_feed})

    print("output shape : ", output.shape)
    # output = scipy.ndimage.zoom(output, 256/25)
    # print(output[0,:,:,1].shape)


    # plt.figure()
    # plt.imshow(output[0], cmap='gray')
    
    plt.figure()
    plt.imshow(output[0,:,:,1], cmap=plt.cm.rainbow)

    # plt.figure()
    # plt.imshow(output[0,:,:,0], cmap='gray')

    plt.show()
        
        
    
        
        
    #     print('truth : ', int(targets[idx]), 'predicted : ', pred, "probs : ", soft)
    #     if int(targets[idx]) == pred[0][0][0]:
    #         correct += 1

    # print("Total : ", len(features), "correct : ", correct)
    # print("Accuracy : ", (correct/len(features))*100.0 , "%")
    



