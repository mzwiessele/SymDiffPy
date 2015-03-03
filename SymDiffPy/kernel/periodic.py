import GPy, numpy as np
import theano, theano.tensor as T
from theano.tensor import slinalg, nlinalg


class PeriodicExp(GPy.core.Parameterized):
    def __init__(self, input_dim, alpha=None, lengthscale=None, period=None, ARD=False, PeriodPerDimension=False, name='periodicexponential'):
        super(PeriodicExp, self).__init__(name=name)
        if alpha is None:
            alpha = 1.
        self.alpha = GPy.Param('alpha', alpha, GPy.constraints.Logexp())
        
        if lengthscale is None:
            if ARD:
                lengthscale = np.ones(input_dim)
            else:
                lengthscale = np.ones((1,))
        if period is None:
            if PeriodPerDimension:
                period = np.ones(input_dim)
            else:
                period = np.ones((1,))


        self.lengthscale = GPy.Param('lengthscale', lengthscale)
        self.period = GPy.Param('period', period, GPy.constraints.Logexp())        
        print self.lengthscale.values
        print self.period.values        
        print self.alpha.values
        self.link_parameters(self.alpha, self.lengthscale, self.period)
        
        self.T_alpha = T.dvector('alpha')
        if ARD:
            self.T_lengthscale = T.TensorType('float64', (False,))(name='lengthscale')
            #self.T_lengthscale = T.dvector('lengthscale')
        else:
            self.T_lengthscale = T.TensorType('float64', (True,))(name='lengthscale')

        if PeriodPerDimension:
            self.T_period = T.TensorType('float64', (False,))(name='period')
            #self.T_lengthscale = T.dvector('lengthscale')
        else:
            self.T_period = T.TensorType('float64', (True,))(name='period')

            
        #self.T_period = T.dvector('period')
        self.T_list = [self.T_alpha, self.T_lengthscale, self.T_period]# must be defined to be used in upper classes
        
    def init_theano(self, X1, X2):
        self.T_X1 = X1
        self.T_X2 = X2
        #self.T_K = self.T_alpha[0] * T.exp(-.5*T.sum(T.sqr((self.T_X1[:, None, :]-self.T_X2[None, :, :])/self.T_lengthscale), -1))
        self.T_K = self.T_alpha[0] * T.exp(-2.0*T.sum(T.sqr(T.sin((np.pi*T.abs_((self.T_X1[:, None, :]-self.T_X2[None, :, :])))/self.T_period[None,None,:])/self.T_lengthscale[None,None,:]),-1))
        self.f_K = theano.function([self.T_X1, self.T_X2, self.T_alpha, self.T_lengthscale, self.T_period], self.T_K)
    
    def K(self, X1, X2=None):
        if X2 is None:
            return self.f_K(X1, X1, self.alpha, self.lengthscale, self.period)
        else:
            return self.f_K(X1, X2, self.alpha, self.lengthscale, self.period)
