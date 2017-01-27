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

def construct_mnist_mlp(input_var, no_hid_layers, in_dropout, hid_dropout, alpha, eps):
    # input layer
    mlp = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    mlp = xnor_net.DenseLayer(
        mlp, 
        xnor=True,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=4096)

    for h in range(no_hid_layers-1):

        mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=eps, 
            alpha=alpha)
        mlp = xnor_net.DenseLayer(
            mlp, 
            xnor=True,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=4096)

        mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=hid_dropout)


    mlp = xnor_net.DenseLayer(
        mlp, 
        xnor=False,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=10)
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=eps, 
            alpha=alpha)

    return mlp

def construct_mnist_convnet(input_var):
    cnn = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    cnn = xnor_net.Conv2DLayer(
        cnn,
        xnor=True,
        num_filters=20, 
        filter_size=(5, 5),
        pad=2,
        nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = xnor_net.Conv2DLayer(
        cnn,
        xnor=True,
        num_filters=50, 
        filter_size=(5, 5),
        pad=2,
        nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = xnor_net.DenseLayer(
        cnn,
        xnor=True,
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=500)
    cnn = lasagne.layers.NonlinearityLayer(
        cnn,
        nonlinearity=lasagne.nonlinearities.rectify)
    cnn = xnor_net.DenseLayer(
            cnn,
            xnor=False,
            nonlinearity=lasagne.nonlinearities.softmax,
            #nonlinearity=lasagne.nonlinearities.identity,
            num_units=10)
    return cnn

if __name__=='__main__':

    # This is XNOR net
    xnor = True

    # type of loss
    softmax = False

    # Model file name
    model_file = 'mnist_allxnor_mlp.npz'

    # hyper parameters
    batch_size = 100
    alpha = 0.1
    eps = 1e-4
    no_epochs = 1000

    # dropout
    input_dropout = 0.2
    hidden_dropout = 0.5

    no_hid_layers = 3

    # learning rate
    # similar settings as in BinaryNet
    LR_start = 0.003
    LR_end = 0.0000003
    LR_decay = (LR_end/LR_start)**(1./no_epochs)
    print('LR_start = {:f}\tLR_end = {:f}\tLR_decay = {:f}'.format(LR_start, LR_end, LR_decay))

    # input data, target and learning rate as theano symbolic var
    input_vars = T.tensor4('input')
    targets = T.fmatrix('target')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    # construct deep network
    print('Constructing the network...')
    net = construct_mnist_mlp(input_vars, no_hid_layers, input_dropout, hidden_dropout, alpha, eps)
    #net = construct_mnist_convnet(input_vars)

    # Load data
    print('Loading the data...')
    train_x, val_x, test_x, train_y, val_y, test_y = cnn_utils.load_data('mnist')

    if(not softmax):
        # for hinge loss
        train_y = 2* train_y - 1.
        val_y = 2* val_y - 1.
        test_y = 2* test_y - 1.

    # network output
    print('Constructed symbolic output')
    train_pred = lasagne.layers.get_output(net, deterministic=False)

    # loss. As per paper it is -ve log-liklihood on softmax output
    if(softmax):
        loss = lasagne.objectives.categorical_crossentropy(train_pred, targets)
        # mean loss across all images in the batch
        loss = T.mean(loss)
    else: #hinge loss
        loss = T.mean(T.sqr(T.maximum(0.,1.-targets*train_pred)))

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
    if(softmax):
        test_loss = T.mean(lasagne.objectives.categorical_crossentropy(test_pred, targets))
    else:
        test_loss = T.mean(T.sqr(T.maximum(0.,1.-targets*test_pred)))
    test_err = T.mean(T.neq(T.argmax(test_pred, axis=1), T.argmax(targets, axis=1)),dtype=theano.config.floatX)

    # construct theano function train, validation/testing process
    train_fn = theano.function([input_vars, targets, LR], loss, updates=updates)

    #test_fn = theano.function([input_vars, targets], test_loss)
    test_fn = theano.function([input_vars, targets], [test_loss, test_err])
    print('Created theano functions for training and validation...')

    print('Training...')
    print('Trainset shape = ', train_x.shape, train_y.shape)
    print('Valset shape = ', val_x.shape, val_y.shape)
    print('Testset shape = ', test_x.shape, test_y.shape)
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


   # achieves around 3% error rate 
