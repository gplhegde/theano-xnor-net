import sys, os, time
import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle
import xnor_net
import cnn_utils
from external import bnn_utils
import gzip
from collections import OrderedDict

def construct_cifar10_net(input_var, alpha, eps):
    # input layer
    cnn = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # Input conv layer is not binary. As the paper states, the computational savings are very less
    # when the input channels to the conv layer are less
    cnn = xnor_net.Conv2DLayer(
        cnn,
        xnor=False,
        num_filters=128, 
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.Conv2DLayer(
            cnn, 
            xnor=True,
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.Conv2DLayer(
            cnn, 
            xnor=True,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.Conv2DLayer(
            cnn, 
            xnor=True,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.Conv2DLayer(
            cnn, 
            xnor=True,
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.Conv2DLayer(
            cnn, 
            xnor=True,
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.DenseLayer(
            cnn, 
            xnor=True,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.DenseLayer(
            cnn, 
            xnor=True,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=eps, 
            alpha=alpha)

    cnn = xnor_net.DenseLayer(
            cnn, 
            xnor=False,
            nonlinearity=lasagne.nonlinearities.softmax,
            num_units=10)

    return cnn

if __name__=='__main__':

    # This is XNOR net
    xnor = True
    # Model file name
    model_file = 'xnor_net_cifar10_nonxnor_first_lyr.npz'

    # hyper parameters
    batch_size = 50
    alpha = 0.1
    eps = 1e-4
    no_epochs = 200
    # learning rate
    # similar settings as in BinaryNet
    LR_start = 0.001
    LR_end = 0.0000003
    LR_decay = (LR_end/LR_start)**(1./no_epochs)
    print('LR_start = {:f}\tLR_end = {:f}\tLR_decay = {:f}'.format(LR_start, LR_end, LR_decay))

    # input data, target and learning rate as theano symbolic var
    input_vars = T.tensor4('input')
    targets = T.fmatrix('target')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    # construct deep network
    print('Constructing the network...')
    net = construct_cifar10_net(input_vars, alpha, eps)

    # Load data
    print('Loading the data...')
    train_x, val_x, test_x, train_y, val_y, test_y = cnn_utils.load_data('cifar10')

    # network output
    print('Constructed symbolic output')
    train_pred = lasagne.layers.get_output(net, deterministic=False)

    # loss. As per paper it is -ve log-liklihood on softmax output
    loss = lasagne.objectives.categorical_crossentropy(train_pred, targets)
    # mean loss across all images in the batch
    loss = T.mean(loss)

    print('Constructed symbolic training loss')

    # define the update process. No need of weight cliping as in BinaryNet
    print('Defining the update process...')
    if xnor:
        
        # W updates
        W = lasagne.layers.get_all_params(net, xnor=True)
        W_grads = bnn_utils.compute_grads(loss,net)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = bnn_utils.clipping_scaling(updates, net)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(net, trainable=True, xnor=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, 
            params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(net, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    # test prediction and loss expressions
    print('Creating test prediction, loss and error expressions...')
    test_pred = lasagne.layers.get_output(net, deterministic=True)
    test_loss = T.mean(lasagne.objectives.categorical_crossentropy(test_pred, targets))
    test_err = T.mean(T.neq(T.argmax(test_pred, axis=1), T.argmax(targets, axis=1)),dtype=theano.config.floatX)

    # construct theano function train, validation/testing process
    train_fn = theano.function([input_vars, targets, LR], loss, updates=updates)

    #test_fn = theano.function([input_vars, targets], test_loss)
    test_fn = theano.function([input_vars, targets], [test_loss, test_err])
    print('Created theano functions for training and validation...')

    print('Training...')
    #new_loss = train_fn(train_x[0:50], train_y[0:50], LR_start)
    print('Trainset shape = ', train_x.shape, train_y.shape)
    print('Valset shape = ', val_x.shape, val_y.shape)
    print('Testset shape = ', test_x.shape, test_y.shape)
    #new_loss, new_err = test_fn(val_x[0:50], val_y[0:50])
    bnn_utils.train(
            train_fn,test_fn,
            net,
            batch_size,
            LR_start,LR_decay,
            no_epochs,
            train_x,train_y,
            val_x,val_y,
            test_x,test_y,
            save_path=model_file,
            shuffle_parts=1)

# This should produce at most 13.89% test error rate

    
