import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg

class RBF(GPy.core.Parameterized):
    def __init__(self, input_dim, alpha=None, lengthscale=None, ARD=False, name='rbf'):
        super(RBF, self).__init__(name=name)
        if alpha is None:
            alpha = 1.
        self.alpha = GPy.Param('alpha', alpha, GPy.constraints.Logexp())
        if lengthscale is None:
            if ARD:
                lengthscale = np.ones(input_dim)
            else:
                lengthscale = 1.
        self.lengthscale = GPy.Param('lengthscale', lengthscale)
        self.link_parameters(self.alpha, self.lengthscale)
        
        self.T_alpha = T.dvector('alpha')
        self.T_lengthscale = T.dvector('lengthscale')
        self.T_list = [self.T_alpha, self.T_lengthscale]# must be defined to be used in upper classes
        
    def init_theano(self, X1, X2):
        self.T_X1 = X1
        self.T_X2 = X2
        self.T_K = self.T_alpha[0] * T.exp(-.5*T.sum(T.sqr((self.T_X1[:, None, :]-self.T_X2[None, :, :])/self.T_lengthscale), -1))
        self.f_K = theano.function([self.T_X1, self.T_X2, self.T_alpha, self.T_lengthscale], self.T_K)
    
    def K(self, X1, X2=None):
        if X2 is None:
            return self.f_K(X1, X1, self.alpha, self.lengthscale)
        else:
            return self.f_K(X1, X2, self.alpha, self.lengthscale)
