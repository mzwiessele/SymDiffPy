import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg

class Linear(GPy.core.Parameterized):
    def __init__(self, input_dim, ARD=False, name='linear'):
        super(Linear, self).__init__(name=name)
        
        self.input_dim = input_dim
        if ARD:
            relevance = np.ones(input_dim)
        else:
            relevance = 1.
    
        self.relevance = GPy.Param('relevance', relevance)
        print self.relevance.values

        self.link_parameters(self.relevance)
        if ARD:
            self.T_relevance = T.TensorType('float64', (False,))('relevance')
        else:
            self.T_relevance = T.TensorType('float64', (True,))('relevance')

        self.T_list = [self.T_relevance]# must be defined to be used in upper classes
        
    def init_theano(self, X1, X2):
        self.T_X1 = X1
        self.T_X2 = X2
        self.T_K = T.dot(self.T_X1, T.dot(T.eye(self.input_dim)*self.T_relevance, self.T_X2.T))
        self.f_K = theano.function([self.T_X1, self.T_X2, self.T_relevance], self.T_K)
    
    def K(self, X1, X2=None):
        if X2 is None:
            return self.f_K(X1, X1, self.relevance)
        else:
            return self.f_K(X1, X2, self.relevance)
