import GPy, numpy as np
import theano.tensor as T, theano
from theano.ifelse import ifelse
from theano.tensor import slinalg, nlinalg

class White(GPy.core.Parameterized):
    def __init__(self, alpha=None, name='white'):
        super(White, self).__init__(name=name)
        if alpha is None:
            alpha = 1.
        self.alpha = GPy.Param('alpha', alpha, GPy.constraints.Logexp())
        
        print self.alpha.values
        self.link_parameters(self.alpha)
        
        self.T_alpha = T.TensorType('float64', (True,))('alpha')
        self.T_list = [self.T_alpha]# must be defined to be used in upper classes
        	
    def init_theano(self, X1, X2):
        self.T_X1 = X1
        self.T_X2 = X2
	self.T_K = T.neq(T.sum(self.T_X1[:, None, :] - self.T_X2[None, :, :],-1),0.) * self.T_alpha
       # self.T_K = ifelse(T.any(T.neq(self.T_X1[:, None, :] - self.T_X2[None, :, :],0.)), self.T_alpha*T.eye(self.T_X1.shape[0]),T.zeros((self.T_X1.shape[0], self.T_X2.shape[0])))
        self.f_K = theano.function([self.T_X1, self.T_X2, self.T_alpha], self.T_K)
    
    def K(self, X1, X2=None):
        if X2 is None:
            return self.f_K(X1, X1, self.alpha)
        else:
            return self.f_K(X1, X2, self.alpha)
