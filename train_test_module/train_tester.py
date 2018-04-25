import os,sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
weight_path = os.path.join(root_path, 'weights', '4cl_SM_epoch167model')
TEST_DATA_PATH =  os.path.join(root_path, "data", "TestData_ALL_COR_5cl.npz")


#Load Test Features and Targets
test_data_load = np.load(TEST_DATA_PATH)
test_features = test_data_load['features']
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[2], test_features.shape[3], test_features.shape[4], test_features.shape[1]))
test_targets = test_data_load['targets']



print(weight_path)


# #Initialize Tensor
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
tf.saved_model.loader.load(sess, ['foo-tag'], weight_path)
tensor_in = tf.get_default_graph().get_tensor_by_name('input_holder:0')
y = tf.get_default_graph().get_tensor_by_name('fc:0')
keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

numNone = 0
aNone = 0

numPartial = 0
aPartial = 0

numSM = 0
aSM = 0

numLarge = 0
aLarge = 0


 # cam_data.append(cam)
for idx, t_target in enumerate(test_targets):
    pred= sess.run(y, feed_dict={tensor_in:[test_features[idx]], keep_prob:1.0})
    print(pred[0])

    if t_target > 1.0: t_target = t_target - 1.0
    
            
    if t_target == 0:
        aNom += 1
    elif t_target == 3:
        numPartial += 1
    elif t_target == 1:
        numSmall += 1
    elif t_target == 2:
        numLarge += 1




    if pred[0] == t_target:
        
        if t_target == 0:
            aNone += 1
        elif t_target == 3:
            aPartial += 1
        elif t_target == 1:
            aSM += 1
        elif t_target == 2:
            aLarge += 1





print("None : ", numNone)
print("Partial : ", numPartial)
print("Small : ", numSmall)
print("Medium : ", numMedium)
print("Large + Massive : ", numLarge)