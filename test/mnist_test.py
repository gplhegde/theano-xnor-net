import sys, os, time
import argparse
import lasagne
import numpy as np
import theano
import theano.tensor as T
import cPickle
import xnornet_layers
import inf_layers
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', 'train'))
import cnn_utils
from  fxp_helper import analyze_param_range, convert_fxp_format, fixed_point
from inf_utils import set_network_params


def parse_args():
    """Argument parser for this script
    """
    parser = argparse.ArgumentParser(description='Test MNIST classification performance using XNOR-Net')
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

def construct_mnist_mlp(input_var, no_hid_layers, in_dropout, hid_dropout, alpha, eps, data_format='float'):
    # input layer
    data_bits = 15
    fc_int_bits = 10
    norm_int_bits = 3

    mlp = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    mlp = xnornet_layers.DenseLayer(
        mlp, 
        format=data_format,
        data_bits=data_bits,
        int_bits=fc_int_bits, 
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=4096)

    for h in range(no_hid_layers-1):

        mlp = inf_layers.BatchNormLayer(
            mlp,
            format=data_format,
            data_bits=data_bits,
            int_bits=norm_int_bits,
            epsilon=eps, 
            alpha=alpha)
        mlp = xnornet_layers.DenseLayer(
            mlp,
            format=data_format,
            data_bits=data_bits,
            int_bits=fc_int_bits, 
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=4096)
        # TODO: dropout layer is not needed for inference. remove it later
        mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=hid_dropout)

    # TODO: The input to this dense layer are too large because there is not norm layer immediately below this.
    # Hence the no of integer bits required for this layer is quite large compared to xnor-dense layers.
    # Try to train with one more norm layer below this and see if that helps to cut down the integer bits.
    mlp = inf_layers.DenseLayer(
        mlp, 
        format=data_format,
        data_bits=20,
        int_bits=17, 
        nonlinearity=lasagne.nonlinearities.identity,
        num_units=10)
    mlp = inf_layers.BatchNormLayer(
            mlp,
            format=data_format,
            data_bits=data_bits,
            int_bits=norm_int_bits,
            epsilon=eps, 
            alpha=alpha)

    return mlp

             
def test_mnist(model, no_imgs, arith_format):
    # This is XNOR net
    xnor = True

    # type of loss
    softmax = False

    # dropout
    input_dropout = 0.2
    hidden_dropout = 0.5

    alpha = 0.1
    eps = 1e-4

    no_hid_layers = 3

    # input data, target and learning rate as theano symbolic var
    input_vars = T.tensor4('input')
    targets = T.fmatrix('target')

    # construct deep network
    print('Constructing the network...')
    net = construct_mnist_mlp(input_vars, no_hid_layers, input_dropout, hidden_dropout, alpha, eps, arith_format)

    # Load data
    print('Loading the data...')
    train_x, val_x, test_x, train_y, val_y, test_y = cnn_utils.load_data('mnist')

    if(no_imgs > len(test_x)):
        print('Max available test images = {:d}'.format(len(test_x)))
        print('Testing with max number of available test images')
        no_imgs = len(test_x)

    # range conversion for hinge loss
    test_y = 2* test_y - 1.

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

    # first dense layer is xnor, hence start from param[0]
    # last 2 layers are non-xnor dense layer + normalization which have 2 and 6 params respectively. Hence end at -8
    for param in params[0:-8]:
        # print param.name
        if param.name == "W":
            param.set_value(xnornet_layers.SignNumpy(param.get_value()))

    if(arith_format == 'fixed'):
        print('Using FIXED point mode for testing...')
        # fixced point # of bits excluding sign bit
        total_bits = 15
        convert_fxp_format(lasagne.layers.get_all_params(net), total_bits)
        test_x = fixed_point(test_x, 7, 0)

    # start testing
    print('Testing {:d} images from the MNIST test set....'.format(no_imgs))
    start_time = time.time()

    error = test_fn(test_x[0:no_imgs], test_y[0:no_imgs]) * 100
    runtime = time.time() - start_time
    print('Testing Accuracy = {:f}%'.format(100 - error))
    print('Test time = {:f} seconds'.format(runtime))

 
if __name__=='__main__':
    args = parse_args()

    test_mnist(args.model_file, args.no_imgs, args.mode)
