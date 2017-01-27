import sys, os, time
import argparse
import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle, time
import inf_layers
from fxp_helper import convert_fxp_format, fixed_point
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'train'))
import cnn_utils
import gzip
from collections import OrderedDict
import xnornet_layers
from inf_utils import set_network_params


def parse_args():
    """Argument parser for this script
    """
    parser = argparse.ArgumentParser(description='Test CIFAR-10 classification performance using XNOR-Net')
    parser.add_argument('--model', dest='model_file', help='XNOR-Net trained model file in .npz format')
    parser.add_argument('--no', dest='no_imgs', type=int, help='Number of images to test. Max = 10000')
    parser.add_argument('--mode', dest='mode', default='float', choices=['fixed', 'float'],
        help='Arithmetic mode, default = float')

    # parse command line args
    if(len(sys.argv) < 5):
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

def construct_cifar10_testnet(input_var, data_format='float'):
    data_bits = 15
    conv_int_bits = 8
    norm_int_bits = 3
    fc_int_bits = 10

    print('Constructing the network...')
    # input layer
    cnn = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # Input conv layer is not binary. As the paper states, the computational savings are very less
    # when the input channels to the conv layer are less
    cnn = inf_layers.Conv2DLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=conv_int_bits,
        num_filters=128, 
        filter_size=(3, 3),
        pad=1,
        nonlinearity=lasagne.nonlinearities.identity)

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.Conv2DLayer(
            cnn,
            format=data_format,
            data_bits=data_bits,
            int_bits=conv_int_bits,
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)


    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.Conv2DLayer(
            cnn, 
            format=data_format,
            data_bits=data_bits,
            int_bits=conv_int_bits,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.Conv2DLayer(
            cnn, 
            format=data_format,
            data_bits=data_bits,
            int_bits=conv_int_bits,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)


    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.Conv2DLayer(
            cnn, 
            format=data_format,
            data_bits=data_bits,
            int_bits=conv_int_bits,
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.Conv2DLayer(
            cnn, 
            format=data_format,
            data_bits=data_bits,
            int_bits=conv_int_bits,
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.DenseLayer(
            cnn, 
            format=data_format,
            data_bits=data_bits,
            int_bits=fc_int_bits,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = xnornet_layers.DenseLayer(
            cnn,
            format=data_format,
            data_bits=data_bits,
            int_bits=fc_int_bits,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)

    cnn = inf_layers.BatchNormLayer(
        cnn,
        format=data_format,
        data_bits=data_bits,
        int_bits=norm_int_bits)

    cnn = inf_layers.DenseLayer(
            cnn,
            format=data_format,
            data_bits=data_bits,
            int_bits=fc_int_bits, 
            nonlinearity=lasagne.nonlinearities.softmax,
            num_units=10)

    return cnn

def test_cifar(model, no_imgs, arith_format):


    # input data, target and learning rate as theano symbolic var
    input_vars = T.tensor4('input')
    targets = T.fmatrix('target')

    # construct deep network
    print('Constructing the network...')
    net = construct_cifar10_testnet(input_vars, arith_format)

    # Load data
    print('Loading the data...')
    train_x, val_x, test_x, train_y, val_y, test_y = cnn_utils.load_data('cifar10')

    if(no_imgs > len(test_x)):
        print('Max available test images = {:d}'.format(len(test_x)))
        print('Testing with max number of available test images')
        no_imgs = len(test_x)

    # test prediction and loss expressions
    print('Creating test prediction, loss and error expressions...')
    test_pred = lasagne.layers.get_output(net, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_pred, axis=1), T.argmax(targets, axis=1)),dtype=theano.config.floatX)

    test_fn = theano.function([input_vars, targets], test_err)
   
    print('Initializing the model parameters...')
    with np.load(model) as mf:
        params = [mf['arr_{:d}'.format(i)] for i in range(len(mf.files))]

    set_network_params(net, params)

    # Binarize the weights. The weight scaling factors are already part of the model.
    # No need to compute them again
    params = lasagne.layers.get_all_params(net)
    # first conv layer and last dense dense layer which has 2 parameters is not xnor. 
    #Hence leave first(W) last 2 params(W, b) from binarization
    for param in params[1:-2]:
        if param.name == "W":
            param.set_value(xnornet_layers.SignNumpy(param.get_value()))

    if(arith_format == 'fixed'):
        print('Using FIXED point mode for testing...')
        # fixced point # of bits excluding sign bit for all parameters
        param_total_bits = 15
        convert_fxp_format(lasagne.layers.get_all_params(net), param_total_bits)
        # input data is in the range [-1 +1] use 7 bits for magnitude (1 bit for sign)
        test_x = fixed_point(test_x, 7, 0)

    # start testing
    print('Testing {:d} images'.format(no_imgs))
    start_time = time.time()
    error_batch = test_fn(test_x[0:no_imgs], test_y[0:no_imgs]) * 100
    runtime = time.time() - start_time
    print('Testing Accuracy = {:f}%'.format(100 - error_batch))
    print('Test time = {:f} seconds'.format(runtime))

if __name__=='__main__':
    args = parse_args()

    test_cifar(args.model_file, args.no_imgs, args.mode) 
