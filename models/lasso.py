import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg

from matplotlib import pyplot as plt


class Lasso(GPy.core.Parameterized):
    def __init__(self, X, Y, l, beta=None, name="lasso"):
        super(Lasso, self).__init__(name=name)
        self.X = X
        self.Y = Y
        self.l = l
        
        if beta is None:
            beta = np.zeros((self.X.shape[1], self.Y.shape[1]))
        
        self.beta = GPy.core.parameterization.Param("beta", beta)
        self.link_parameter(self.beta)
        
        # Theano init
        self.T_beta = T.dmatrix('beta')
        self.T_X = T.dmatrix('X')
        self.T_Y = T.dmatrix('Y')
        
        self.T_obj = T.sum(T.sqr(self.T_Y - T.dot(self.T_X, self.T_beta))) + T.sum(T.abs_(self.l * self.T_beta))
        self.T_grad = theano.grad(self.T_obj, self.T_beta)

        self.f_obj = theano.function([self.T_Y, self.T_X, self.T_beta], self.T_obj)
        self.f_grad = theano.function([self.T_Y, self.T_X, self.T_beta], self.T_grad)

    def log_likelihood(self):
        return -self._obj
        
    def parameters_changed(self):
        self._obj = self.f_obj(self.Y, self.X, self.beta)
        self.beta.gradient[:] = -self.f_grad(self.Y, self.X, self.beta)
