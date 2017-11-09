import logging
import math
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

#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.insert(0, root_path)
sys.setrecursionlimit(2000)

import network.VRN_64_gpuarray as config_module
DATA_PATH = os.path.join(root_path, "data/TestData.npz")
WEIGHT_NAME = "10-20 08:28:29gpuarray_oct_20_epoch_1.npz"
WEIGHT_PATH = os.path.join(root_path, 'weights', WEIGHT_NAME)

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
    softmax = T.nnet.softmax(T.sum(y_hat_deterministic, axis=0))

    #Get Annotation
    anot = T.mean(y,dtype='int32')

    predFunc = theano.function([X], pred)


    # Get error rate
    classifier_test_error_rate = T.cast( T.mean( T.neq(pred, anot)), 'float32' )



    # Compile Functions
    test_error_fn = theano.function([batch_index], [classifier_test_error_rate,pred, anot, softmax], givens={
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


def test_accuracy(weight_path, tfuncs, tvars,model, predFunc):
    
    print(weight_path)

    # Load weights
    metadata = checkpoints.load_weights(weight_path, model['l_out'])

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
    num_test_chunks=int(math.ceil(float(len(xt))/test_chunk_size))
    print("num_test_chunks", num_test_chunks)
    num_test_batches = int(math.ceil(float(len(xt))/float(n_rotations)))
    print(num_test_batches)

    # Prepare test error
    test_class_error = []

    # Initialize test iteration counter
    test_itr=0

    correct_features = []
    correct_targets = []

    score = []
    y = []


    # Loop across chunks!
    for chunk_index in range(num_test_chunks):

        # Define upper index of chunk
        upper_range = min(len(yt),(chunk_index+1)*test_chunk_size)

        # Get chunks
        x_shared = np.asarray(xt[chunk_index*test_chunk_size:upper_range,:,:,:,:],dtype=np.float32)
        y_shared = np.asarray(yt[chunk_index*test_chunk_size:upper_range],dtype=np.float32)

        # Get number of batches for this chunk
        num_batches = len(x_shared)//n_rotations

        # Prepare dat
        # tvars['X_shared'].set_value(4.0 * x_shared-1.0, borrow=True)
        tvars['X_shared'].set_value(x_shared, borrow=True)
        tvars['y_shared'].set_value(y_shared, borrow=True)

        # Loop across batches!
        for bi in range(num_batches):

            # Increment test iteration counter
            test_itr+=1

            # Test!
            [batch_test_class_error,pred, anot, softmax] = tfuncs['test_function'](bi)

            # Record test results
            test_class_error.append(batch_test_class_error)

            

            # if batch_test_class_error == 1.0:
            print(test_itr, "-> Input : ", anot , " // pred : " , pred , " // neq : " , batch_test_class_error, softmax)


            
            y.append(int(anot))
            score.append(softmax[0][1])
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
    

  
    
    np.savez_compressed(os.path.join(file_path, 'roc_data', WEIGHT_NAME[:-4]+'roc'), y=y, score=score)
    # plt.plot(roc_x, roc_y, 'b-')
    # plt.show()


# Get Model

print('Compiling theano functions...')
cfg = config_module.cfg
model = config_module.get_model()
tfuncs, tvars,model, predFunc = make_testing_functions(cfg,model)
test_accuracy(WEIGHT_PATH, tfuncs, tvars,model, predFunc)