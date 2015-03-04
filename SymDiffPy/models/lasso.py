import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg

from matplotlib import pyplot as plt

class Lasso(GPy.core.model.Model):
    def __init__(self, X, Y, l, beta=None, intercept=True, name="lasso"):
        super(Lasso, self).__init__(name=name)
        self.X = X
        self.Y = Y
        self.l = l
        self.num_data, self.output_dim, self.input_dim = self.X.shape[0], self.Y.shape[1], self.X.shape[1]        
        self.intercept=intercept
        if intercept:
            self.X = np.c_[np.ones(X.shape[0]),X]
            

        
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
        
    def predict(self,Xnew):
        return np.dot(Xnew, self.beta),0
        
    def plot(self, resolution=100):
        if self.input_dim == 1:
            plt.scatter(self.X[:,int(self.intercept)],self.Y,color='k',marker='x')
            mi, ma = self.X.min(), self.X.max()
            ten_perc = .1*(ma-mi)
            Xpred = np.linspace(mi-ten_perc, ma+ten_perc, resolution)[:,None]
            if self.intercept:
                Xpred = np.c_[np.ones(Xpred.shape[0]),Xpred]
            mu, var = self.predict(Xpred)
            plt.plot(Xpred[:,int(self.intercept)], mu, color='g', lw=1.5)
            #plt.fill_between(Xpred[:, 0], mu[:,0]+2*np.sqrt(np.diagonal(var)), mu[:,0]-2*np.sqrt(np.diagonal(var)), color='k', alpha=.1)
        else:
            raise NotImplementedError("Only one dim plots allowed")

# class Lasso(GPy.core.Parameterized):
#     def __init__(self, X, Y, l, beta=None, name="lasso"):
#         super(Lasso, self).__init__(name=name)
#         self.X = X
#         self.Y = Y
#         self.l = l
        
#         if beta is None:
#             beta = np.zeros((self.X.shape[1], self.Y.shape[1]))
        
#         self.beta = GPy.core.parameterization.Param("beta", beta)
#         self.link_parameter(self.beta)
        
#         # Theano init
#         self.T_beta = T.dmatrix('beta')
#         self.T_X = T.dmatrix('X')
#         self.T_Y = T.dmatrix('Y')
        
#         self.T_obj = T.sum(T.sqr(self.T_Y - T.dot(self.T_X, self.T_beta))) + T.sum(T.abs_(self.l * self.T_beta))
#         self.T_grad = theano.grad(self.T_obj, self.T_beta)

#         self.f_obj = theano.function([self.T_Y, self.T_X, self.T_beta], self.T_obj)
#         self.f_grad = theano.function([self.T_Y, self.T_X, self.T_beta], self.T_grad)

#     def log_likelihood(self):
#         return -self._obj
        
#     def parameters_changed(self):
#         self._obj = self.f_obj(self.Y, self.X, self.beta)
#         self.beta.gradient[:] = -self.f_grad(self.Y, self.X, self.beta)
