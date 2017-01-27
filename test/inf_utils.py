import lasagne
import theano
import numpy as np

def set_network_params(net, param_vals):
    """This method will set all parameters of the network using trained param values
    The Batch Norm Layer of the inference network contains two extra parameters which are 
    nothing but combinations of the original 4 parameters(mean, gamma, beta, inv_std).
    Hence they are not used during training. This method will derive these 2 extra params and
    set them in the test network
    """
    params = lasagne.layers.get_all_params(net)
    pidx = 0
    for p in params:
        if(p.name == 'sub'):
            # these are clubbed params of norm layer.
            #print('Setting clubbed offset param for norm layer')
            inv_std = param_vals[pidx-1]
            mean = param_vals[pidx-2]
            gamma = param_vals[pidx-3]
            beta = param_vals[pidx-4]
            sub = -mean * inv_std * gamma + beta
            p.set_value(sub)
        elif(p.name == 'scale'):
            #print('Setting clubbed scale param for norm layer')
            inv_std = param_vals[pidx-1]
            gamma = param_vals[pidx-3]
            scale = gamma * inv_std
            p.set_value(scale)
        else:
            p.set_value(param_vals[pidx])
            pidx += 1
