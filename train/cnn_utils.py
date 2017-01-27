import numpy as np
import cPickle
import cv2
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.svhn import SVHN

_DATASET_SIZE = {'mnist' : 50000, 'cifar10':50000}


def load_data(dataset, train_percent=0.8, val_percent=0.2):
    """ Load MNIST, CIFAR-10 dataset
    dataset: string: MNIST or CIFAR-10
    train_percent: float: percentage of the dataset to be used for training
    val_per: string: percentage of the dataset to be used for the validation purpose
    Output:
    (train_x, val_x, test_x, train_y, val_y, test_y)
    """
    zero_mean = False
    

    if(dataset.lower() == 'mnist'):
        print('Loading MNIST dataset from pylearn2')
        train_set_size = int(_DATASET_SIZE['mnist'] * train_percent)
        train_data = MNIST(which_set='train', start=0, stop=train_set_size, center=zero_mean)
        val_data = MNIST(which_set='train', start=train_set_size, stop=_DATASET_SIZE[dataset], center=zero_mean)
        test_data = MNIST(which_set='test', center=zero_mean)

        # convert labels into 1D array
        train_data.y = np.hstack(train_data.y)
        val_data.y = np.hstack(val_data.y)
        test_data.y = np.hstack(test_data.y)
        # create 10 dimensional vector corresponding to each label
        train_data.y = np.float32(np.eye(10))[train_data.y]
        val_data.y = np.float32(np.eye(10))[val_data.y]
        test_data.y = np.float32(np.eye(10))[test_data.y]

        # TODO: convert the data to range [-1,1]
        # reshape the data into image size(#images, channels, height, width). 
        # Each row contains an image in the original dataset
        train_data.X = np.reshape(train_data.X, (-1, 1, 28, 28))
        val_data.X = np.reshape(val_data.X, (-1, 1, 28, 28))
        test_data.X = np.reshape(test_data.X, (-1, 1, 28, 28))

        # convert to [-1 1] range
        train_data.X = train_data.X * 2.0 - 1.0
        val_data.X = val_data.X * 2.0 - 1.0
        test_data.X = test_data.X * 2.0 - 1.0
       

    elif(dataset.lower() == 'cifar10'):
        print('Loading CIFAR-10 dataset from pylearn2')
        train_set_size = int(_DATASET_SIZE['cifar10'] * train_percent)
        train_data = CIFAR10(which_set='train', start=0, stop=train_set_size)
        val_data = CIFAR10(which_set='train', start=train_set_size, stop=50000)
        test_data = CIFAR10(which_set='test')

        # convert labels into 1D array
        train_data.y = np.hstack(train_data.y)
        val_data.y = np.hstack(val_data.y)
        test_data.y = np.hstack(test_data.y)
        # create 10 dimensional vector corresponding to each label
        train_data.y = np.float32(np.eye(10))[train_data.y]
        val_data.y = np.float32(np.eye(10))[val_data.y]
        test_data.y = np.float32(np.eye(10))[test_data.y]

        # TODO: convert the data to range [-1,1]
        # reshape the data into image size(#images, channels, height, width). 
        # Each row contains an image in the original dataset
        train_data.X = np.reshape(train_data.X, (-1, 3, 32, 32))
        val_data.X = np.reshape(val_data.X, (-1, 3, 32, 32))
        test_data.X = np.reshape(test_data.X, (-1, 3, 32, 32))

        # convert to [-1 1] range
        train_data.X = train_data.X * (2.0/255) - 1.0
        val_data.X = val_data.X * (2.0/255) - 1.0
        test_data.X = test_data.X * (2.0/255) - 1.0
    elif(dataset.lower() == 'svhn'):
        train_data = SVHN(which_set= 'splitted_train', axes= ['b', 'c', 0, 1])     
        val_data = SVHN(which_set= 'valid', axes= ['b', 'c', 0, 1])    
        test_data = SVHN(which_set= 'test', axes= ['b', 'c', 0, 1])
        # convert labels into 1D array
        train_data.y = np.hstack(train_data.y)
        val_data.y = np.hstack(val_data.y)
        test_data.y = np.hstack(test_data.y)
        # create 10 dimensional vector corresponding to each label
        train_data.y = np.float32(np.eye(10))[train_data.y]
        val_data.y = np.float32(np.eye(10))[val_data.y]
        test_data.y = np.float32(np.eye(10))[test_data.y]
        # convert to [-1, 1] range
        train_data.X = np.reshape(np.subtract(np.multiply(2.0/255, train_data.X), 1.0), (-1, 3, 32, 32))
        val_data.X = np.reshape(np.subtract(np.multiply(2.0/255, val_data.X), 1.0), (-1, 3, 32, 32))
        test_data.X = np.reshape(np.subtract(np.multiply(2.0/255, test_data.X), 1.0), (-1, 3, 32, 32))
    else:
        print('This dataset is not supported. Only MNIST and CIFAR-10 are supported as of now.')
        raise ValueError('Dataset is not supported')

    print('Trainset shape = ', train_data.X.shape, train_data.y.shape)
    print('Valset shape = ', val_data.X.shape, val_data.y.shape)
    print('Testset shape = ', test_data.X.shape, test_data.y.shape)
    return train_data.X, val_data.X, test_data.X, train_data.y, val_data.y, test_data.y


if __name__=='__main__':
    dataset = 'mnist'
    load_data(dataset)


