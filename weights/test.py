import matplotlib.pyplot as plt




###
# Discriminative Voxel-Based ConvNet Training Function
# A Brock, 2016

import imp
import time
import logging
import math
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
# import theano.sandbox.cuda.basic_ops as sbcuda
import lasagne

import sys, os
sys.path.insert(0, os.getcwd())
from utils import checkpoints, metrics_logging
from collections import OrderedDict
import matplotlib

CONFIG_PATH = '/home/ej/projects/EJ_ROTATORCUFF_CLASSIFIER/network/VRN_64_gpuarray.py'
DATA_PATH = '/home/ej/projects/EJ_ROTATORCUFF_CLASSIFIER/data/TestData.npz'


# Define the testing functions
def make_testing_functions(cfg,model):

    # Input Array
    X = T.TensorType('float32', [False]*5)('X')

    # Shared Variable for input array
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')

    # Class Vector
    y = T.TensorType('int32', [False]*1)('y')

    # Shared Variable for class vector
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    # Output layer
    l_out = model['l_out']

    # Batch Parameters
    batch_index = T.iscalar('batch_index')
    test_batch_slice = slice(batch_index*cfg['n_rotations'], (batch_index+1)*cfg['n_rotations'])
    test_batch_i = batch_index

    # Get output
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)

    # Average across rotation examples
    pred = T.argmax(T.sum(y_hat_deterministic,axis=0))

    #Get Annotation
    anot = T.mean(y,dtype='int32')

    predFunc = theano.function([X], pred)


    # Get error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, anot)), 'float32' )



    # Compile Functions
    test_error_fn = theano.function([batch_index], [classifier_test_error_rate,pred, anot], givens={
            X: X_shared[test_batch_slice],
            y:  T.cast( y_shared[test_batch_slice], 'int32')
        })
    tfuncs = {'test_function':test_error_fn}
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
            }
    return tfuncs, tvars, model, predFunc

listFile = list(glob.iglob('/home/ej/projects/EJ_ROTATORCUFF_CLASSIFIER/train-test_module/tmp/*.npz', recursive = False))
listFile.sort()

xData = []
yData = []
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('on')
ax.set_title('title')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.grid(True)
for idx, WEIGHT_PATH in enumerate(listFile):
    print(WEIGHT_PATH, idx)

    # Load config module
    # Compile functions
    config_module = __import__('VRN_64_gpuarray', CONFIG_PATH[:-3])
    cfg = config_module.cfg

    # Find weights file
    weights_fname = WEIGHT_PATH

    # Get Model
    model = config_module.get_model()

    print('Compiling theano functions...')
    tfuncs, tvars,model, predFunc = make_testing_functions(cfg,model)

    # Load weights
    metadata = checkpoints.load_weights(weights_fname, model['l_out'])

    # Check if previous best accuracy is in metadata from previous tests
    best_acc = metadata['best_acc'] if 'best_acc' in metadata else 0
    print('best accuracy = '+str(best_acc))


    print('Testing...')
    itr = 0

    # Load testing data into memory
    xt = np.asarray(np.load(DATA_PATH)['features'],dtype=np.float32)
    yt = np.asarray(np.load(DATA_PATH)['targets'],dtype=np.float32)

    print(xt.shape[0])
    print(yt.shape)

    n_rotations = cfg['n_rotations']
    print("n_rotations", n_rotations)


    # Determine chunk size
    test_chunk_size = n_rotations*cfg['batches_per_chunk']
    print("test chunk size", test_chunk_size)
    # Determine number of chunks
    num_test_chunks=int(math.ceil(float(len(xt))/test_chunk_size))
    print("num_test_chunks", num_test_chunks)
    # Total number of test batches. Note that we're treating all the rotations
    # of a single instance as a single batch. There's definitely a more efficient
    # way to do this, and you'll want to be more efficient if you implement this in
    # a validation loop, but testing should be infrequent enough that the extra few minutes
    # this costs is irrelevant.
    num_test_batches = int(math.ceil(float(len(xt))/float(n_rotations)))
    print(num_test_batches)

    # Prepare test error
    test_class_error = []

    # Initialize test iteration counter
    test_itr=0

    correct_features = []
    correct_targets = []
    # Loop across chunks!
    for chunk_index in range(num_test_chunks):

        # Define upper index of chunk
        upper_range = min(len(yt),(chunk_index+1)*test_chunk_size)

        # Get chunks
        x_shared = np.asarray(xt[chunk_index*test_chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(yt[chunk_index*test_chunk_size:upper_range],dtype=np.float32)

        # Get number of batches for this chunk
        num_batches = len(x_shared)//n_rotations

        # Prepare data
        # tvars['X_shared'].set_value(4.0 * x_shared-1.0, borrow=True)
        tvars['X_shared'].set_value(x_shared, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)

        # Loop across batches!
        for bi in range(num_batches):

            # Increment test iteration counter
            test_itr+=1

            # Test!
            [batch_test_class_error,pred, anot] = tfuncs['test_function'](bi)

            # Record test results
            test_class_error.append(batch_test_class_error)

            # if batch_test_class_error == 1.0:
            print(test_itr, "-> Input : ", anot , " // pred : " , pred , " // neq : " , batch_test_class_error)

            if int(batch_test_class_error) == 0:
                correct_features.append(x_shared[bi])
                correct_targets.append(y_shared[bi])



            # Optionally, update the confusion matrix
            # confusion_matrix[pred,int(yt[cfg['n_rotations']*test_itr])]+=1

    # Optionally, print confusion matrix
    # print(confusion_matrix)

    # Get total accuracy
    t_class_error = 1-float(np.mean(test_class_error))
    print('Test accuracy is: ' + str(t_class_error))

    


    xData.append(idx)
    yData.append(t_class_error)

    ax.plot(xData, yData, 'ro-')
    fig.canvas.draw()

plt.show(block=True)
