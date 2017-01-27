import theano
import theano.tensor as T
import numpy as np
import lasagne
from fxp_helper import to_fixed_point_theano

# Classic batch normalization layer with fixed point simulation feature for the output data
class BatchNormLayer(lasagne.layers.BatchNormLayer):
    """Class to override the lasagne batch norm layer. This is only implemetend for inference.
    This basically clubs the normalization parameters and pre-multiplies them.
    """
    def __init__(self, incoming, format='float', data_bits=15, int_bits=0, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)



        # club mean, gamma, inv_std
        sub = -self.mean.get_value() * self.inv_std.get_value() * self.gamma.get_value() + self.beta.get_value()
        scale = self.gamma.get_value() * self.inv_std.get_value()
        # The below 2 params are just combination of layer params used during training.
        # This is just to precompute the data independent products and keep it aside. Also, the range of
        # clubbed params is less compared to isolated params. This helps to reduce the # of integer bits used to represent
        # them in the fixed point
        self.sub = self.add_param(sub, sub.shape, name='sub')
        self.scale = self.add_param(scale, scale.shape, name='scale')
        self.format = format
        # FIXME: The below variables need to be theano shared variables?
        self.data_bits = theano.shared(data_bits)
        self.int_bits = theano.shared(int_bits)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Override the lasagne implementation only during the inference time.
        """
        assert(deterministic), 'This layer is only implemented for inference. Use direct Lasagne implementation during training'
        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        scale = self.scale.dimshuffle(pattern)
        sub = self.sub.dimshuffle(pattern)
        if(self.format == 'fixed'):
            # assuming the parameters are already simulated for respective fixed point formats.
            full_precision_out = input * scale + sub
           
            return to_fixed_point_theano(full_precision_out, self.data_bits, self.int_bits)
        else:
            return input * scale + sub

# Classic Lasagne dense layer with fixed point simulation feature for the output data
class DenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, format='float', data_bits=15, int_bits=0, **kwargs):
        num_inputs = int(np.prod(incoming.output_shape[1:]))
        super(DenseLayer, self).__init__(incoming, num_units,  **kwargs)


        # params for fixed point simulation
        self.format = format
        self.data_bits = theano.shared(data_bits)
        self.int_bits = theano.shared(int_bits)

    def get_output_for(self, input, deterministic=True, **kwargs):
        """ Dense layer with fixed point simulation option
        """
        if(self.format == 'fixed'):
            # dot-product at full precision
            fc_out = super(DenseLayer, self).get_output_for(input, **kwargs)
            # reduce the precision of the output based on the data bit widths specified
            fc_out = to_fixed_point_theano(fc_out, self.data_bits, self.int_bits)
        else:
            fc_out = super(DenseLayer, self).get_output_for(input, **kwargs)

        return fc_out

# Classic Lasagne Conv layer with fixed point simulation feature for the output data
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size, format='float', data_bits=15, int_bits=0, **kwargs):
        
        super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)
        self.format = format
        self.data_bits = data_bits
        self.int_bits = int_bits
    
    def convolve(self, input, deterministic=False, **kwargs):
        feat_maps = super(Conv2DLayer, self).convolve(input, **kwargs)
        if(self.format == 'fixed'):
            feat_maps = to_fixed_point_theano(feat_maps, self.data_bits, self.int_bits)
        else:
            pass
        
        return feat_maps
