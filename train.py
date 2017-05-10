import sys
sys.setrecursionlimit(2000)

import os
import numpy as np
import time
import math
from path import Path

import logging

#Import Theano modules
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
import lasagne
from utils import checkpoints, metrics_logging

#import graph Module
import NetworkData.graph.EJ_VRN as ej_graph

#define root
rootPath = os.path.dirname(os.path.realpath(__file__))


def make_training_functions(cfg, model):
    #Input Array
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')

    #Shared Variable For Input ARray
    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    #output layers
    l_out = model['l_out']


    #batch Parameters
    batch_index = T.iscalar('batch_index')
    batch_slice = slice(batch_index * cfg['batch_size'], (batch_index+1)*cfg['batch_size'])

    #get outputs
    # Get outputs
    y_hat = lasagne.layers.get_output(l_out,X)

    # Get deterministic outputs for validation
    y_hat_deterministic = lasagne.layers.get_output(l_out,X,deterministic=True)

    #define loss functions
    l2_all = lasagne.regularization.regularize_network_params(l_out,lasagne.regularization.l2)

    # Classifier loss function
    classifier_loss = T.cast(T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(y_hat), y)), 'float32')

    # Classifier Error Rate
    classifier_error_rate = T.cast( T.mean( T.neq(T.argmax(y_hat,axis=1), y)), 'float32' )

    # Regularized Loss
    reg_loss = cfg['reg']*l2_all + classifier_loss

    # Get all network params
    params = lasagne.layers.get_all_params(l_out,trainable=True)

        # Handle annealing rate cases
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))


    ##########################
    # Step 3: Define Updates #
    ##########################

    updates=lasagne.updates.nesterov_momentum(reg_loss,params,learning_rate=learning_rate)

    update_iter = theano.function([batch_index], [classifier_loss,classifier_error_rate],
            updates=updates, givens={
            X: X_shared[batch_slice],
            y:  T.cast( y_shared[batch_slice], 'int32')
        })

    tfuncs = {'update_iter':update_iter}
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
             }

    return tfuncs, tvars, model

def jitter_chunk(src, cfg, chunk_index):
    np.random.seed(chunk_index)
    dst = src.copy()

    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst

    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']

    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst


# #Load Train and Target Model

#
# X = np.load(trainData)['features']
# y = np.load(trainData)['targets']
#
# X_test = np.load(testData)['features']
# y_test = np.load(testData)['targets']

print('start training')
lasagne.random.set_rng(np.random.RandomState(1234))

print("load graph configuration")
cfg = ej_graph.cfg

#Set Weights and Metrices Filename
weights_fname = rootPath + "\\NetworkData\\weights\\EJ_VRN.npz"


#Get model
model = ej_graph.get_model()


print('Compiling theano functions...')
tfuncs, tvars,model = make_training_functions(cfg, model)

if os.path.isfile(weights_fname):
    print('load weights')
    metadata = checkpoints.load_weights(weights_fname, model['l_out'])
else:
    print("New Training")



print("Training...")
itr = 0

trainPath =  rootPath + "\\NetworkData\\data\\modelnet40_reshaped_train.npz"
testPath = rootPath + "\\NetworkData\\data\\modelnet40_reshaped_test.npz"


x = np.load(trainPath)['features']
y = np.load(trainPath)['targets']
np.random.seed(42)

#define shuffle indices
index = np.random.permutation(len(x))
print("shuffled index : ", index)

#shuffle inputs
x = x[index]
y = y[index]



#define size of chunk to be loaded into GPU Memory
chunk_size = cfg['batch_size'] * cfg['batches_per_chunk']


#determine number of chunks
num_chunks = int(math.ceil(len(y)/float(chunk_size)))

#Get Current Learning Rate
new_lr = np.float32(tvars['learning_rate'].get_value())

#Loop across training epoches
for epoch in range(cfg['max_epochs']):
    #Tic
    epoch_start_time = time.time()


    #update learning rate
    if isinstance(cfg['learning_rate'], dict) and epoch > 0:
        if any(x==epoch for x in cfg['learning_rate'].keys()):
            lr = np.float32(tvars['learning_rate'].get_value())
            new_lr = cfg['learning_rate'][epoch]
            print("changing learning rate from", lr, "to", new_lr)
            tvars['learning_rate'].set_value(np.float32(new_lr))
        if cfg['decay_rate'] and epoch>0:
            lr = np.float32(tvars['learning_rate'].get_value())
            new_lr = lr*(1-cfg['decay_rate'])
            print("Changing learning rate from ", lr, "to ", new_lr)
            tvars['learning_rate'].set_value(np.float32(new_lr))

        for chunk_index in range(num_chunks):
            upper_range = min(len(y), (chunk_index+1)*chunk_size)

            #Get Current Chunk
            x_shared = np.asarray(x[chunk_index * chunk_size:upper_range, :, :, :, :], dtype=np.float32)
            y_shared = np.asarray(y[chunk_index * chunk_size:upper_range], dtype = np.float32)


            np.random.seed(chunk_index)

            indices = np.random.permutation(2*len(x_shared))

            #get nubmer of batches in this chunk
            num_batches = 2*len(x_shared)//cfg['batch_size']

            #combine data with jittered data, the shuffle and change binary range
            tvars['X_shared'].set_value(4.0 * np.append(x_shared,jitter_chunk(x_shared, cfg,chunk_index),axis=0)[indices]-1.0, borrow=True)
            tvars['y_shared'].set_value(np.append(y_shared,y_shared,axis=0)[indices], borrow=True)



            #prepare loss values
            lvs, accs = [], []

            for bi in range(num_batches):
                #Train!
                [classifier_loss, class_acc] = tfuncs['update_iter'](bi)

                #record batch loss and accuracy
                lvs.append(classifier_loss)
                accs.append(class_acc)


                #update iteration counter
                itr += 1

                #Average Losses and Accuracies across chunk
                [closs, c_acc] = [float(np.mean(lvs)), 1.0-float(np.mean(accs))]

                #report and log loses and accuracies
                print('epoch: {0:^3d}, itr: {1:d}, c_loss: {2:.6f}, class_acc: {3:.5f}'.format(epoch, itr, closs, c_acc))


                            # Every Nth epoch, save weights
            if not (epoch%cfg['checkpoint_every_nth']):
                checkpoints.save_weights(weights_fname, model['l_out'], {'itr': itr, 'ts': time.time(), 'learning_rate': new_lr})

        print('Training done')
