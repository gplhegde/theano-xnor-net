import numpy as np
import theano
import theano.tensor as T
import math

def to_fixed_point_theano(input, no_bits, no_int_bits):
    scale =T.cast(2.**(no_bits - no_int_bits), theano.config.floatX)
    max_val = T.cast((2.**no_bits) - 1, theano.config.floatX)
    scaled = input * scale
    scaled = T.round(scaled)
    scaled = T.clip(scaled, -max_val, max_val)
    return scaled/scale

def to_float_point_theano(input, no_bits, no_int_bits):
    scale =2**(no_bits - no_int_bits)
    return input/scale

def analyze_param_range(params):
    """Method to analyze the range of parameters to help deciding the fixed point format.
    """
    for param in params:
        pval = np.absolute(param.get_value())
        print('Param : {:s}'.format(param.name))
        print('Min abs value = {:f} Max abs value = {:f}'.format(np.amin(pval), np.amax(pval)))

def fixed_point(array, no_mag_bits, no_int_bits):
    """Convert to fixed point and convert it back to float
    """
    factor = 2.0 ** (no_mag_bits - no_int_bits)
    max_val = 2. ** no_mag_bits - 1
    scaled_arr = array * factor
    # round to the nearest value
    scaled_arr = np.around(scaled_arr)
    # saturation
    scaled_arr = np.clip(scaled_arr, -max_val, max_val)
    return scaled_arr/factor
 
def convert_fxp_format(params, total_bits):
    """Simulate fixed point for parameters
    """
    
    for param in params:
        if(param.name not in ('mean', 'beta', 'gamma', 'inv_std', 'W')):
            pval = param.get_value()
            pval_mag = np.absolute(pval)
            max_mag = np.amax(pval_mag)
            no_int_bits = int(math.ceil(math.log(max_mag, 2)))
            # avoid -ve numbers when max_mag < 1
            no_int_bits = max(0, no_int_bits)
            #print('Param name : {:10s}   Max magnitude : {:f} No of integer bits : {:d}'\
            #    .format(param.name, max_mag, no_int_bits))
            assert(total_bits >= no_int_bits), 'Bitwidth is not sufficient'        
            sim_fxp_pval = fixed_point(pval, total_bits, no_int_bits)
            param.set_value(sim_fxp_pval)
        
