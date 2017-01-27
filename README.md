#Theano Implementation of XNOR-Net
------------------------------------------
This is the python based implentation of XNOR-Net([this paper](http://arxiv.org/pdf/1603.05279v3.pdf)) using Theano. New derived layer classes for Lasagne are implemented to support the XNOR-Net Convolution and Fully connected layers. The implementation is used to train and test convnets on MNIST and CIFAR-10 classification tasks. This project is tested on python 2.7. 

#Major dependencies
- Bleeding edge version of Lasagne. Installation instructions [here](https://github.com/Lasagne/Lasagne#installation)
- Bleeding edge version of Pylearn2. Installation instructions [here](http://deeplearning.net/software/pylearn2/#download-and-installation)
- theano, numpy
- Reference datasets (downloading of which is explaained below).

#Steps to download example datasets
- Install pylearn2 as explained in the above link. Set the data path which pylearn2 uses to store the datasets as shown below. You can choose the directory of your choice.
```
export PYLEARN2_DATA_PATH=/opt/lisa/data
```
- Execute below commands to download the MNIST, CIFAR-10 and SVHN  datasets respectively.
```
python <pylearn2 install path>/pylearn2/scripts/datasets/download_mnist.py

bash  <pylearn2 install path>/pylearn2/scripts/datasets/download_cifar10.sh

bash  <pylearn2 install path>/pylearn2/scripts/datasets/download_svhn.sh
```

#Before running
- Make sure *theano.config.floatX* is set to *'float32'*. Refer to the [guidelines to configure theano](http://deeplearning.net/software/theano/library/config.html#environment-variables)
- You can enable GPU mode for faster training. Refer to the same theano configuration guide for enable GPU mode. The training of XNOR-Nets is slower than non-xnor counterparts, because it requires more computations(to binarize the inputs and weights, compute scaling factors and so on...)

#Instructions to run

##Training
To train 3 representative networks performing classification tasks on MNIST, CIFAR-10 and SVHN datsets, run the below commands from this directory.
```
bash ./train/train_mnist.sh

bash ./train/train_cifar.sh

bash ./train/train_svhn.sh
```

The MNIST and CIFAR-10 networks produce around **3.2%** and **13.8%** error rate respectively.
##Testing
The testing of the above representative XNOR-Networks supports two modes - FIXED point and floating point mode. Since the purpose of these networks are embedde classification tasks,  it is more efficient to implement them using FIXED point arithmetic. The scripts under ./test just simualte fixed point mode to see the effect of rounding.

To test the networks that you trained using above commands, run the following commands.
```
python test/mnist_test.py --model <model file path> --no <no of images to test>   --mode <fixed OR float>

python test/cifar10_test.py --model <model file path> --no <no of images to test>   --mode <fixed OR float>
```

The model file will be saved during the training process in the .npz format. Use this model file for the **--model** argument.
The default test mode is **float**ing point . Use **fixed** to enable fixed point.
Note that the 4 different parameters in the batch normalization layer ( mean, variance, gamma, beta) are merged into two parameters ( referred as scale and offset in these scripts). This is to reduce computations as they are constant during the inference.

If you need the trained model for any of the above networks, let me know. Also, please contribute if you manage train XNOR-Nets for different computer vision tasks using this project !

#Misc
## Similar Binary Networks
1. **BinaryNet**

  [Paper](http://arxiv.org/pdf/1602.02830v3.pdf)
  
  [Repo](https://github.com/MatthieuCourbariaux/BinaryNet)
2. **BinaryConnect** 

  [Paper](https://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations.pdf)
