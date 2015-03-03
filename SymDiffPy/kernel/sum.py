import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg
from operator import add

class Sum(GPy.core.Parameterized):
    def __init__(self, kernels, name='sum'):
        super(Sum, self).__init__(name=name)
        self.kernels = kernels
        self.link_parameters(*kernels)
        self._flat_pars = self.flattened_parameters
                
    def init_theano(self, X1, X2):
        self.T_list = []
        self.T_X1 = X1
        self.T_X2 = X2

        for k in self.kernels:
            k.init_theano(X1, X2)
            self.T_list.extend(k.T_list)
        
        self.T_K = reduce(add, [k.T_K for k in self.kernels])
        self.f_K = theano.function([self.T_X1, self.T_X2] + self.T_list, self.T_K)
    
    def K(self, X1, X2=None):
        if X2 is None:
            return self.f_K(X1, X1, *self._flat_pars)
        else:
            return self.f_K(X1, X2, *self._flat_pars)
