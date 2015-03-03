import GPy, numpy as np
import theano.tensor as T, theano
from theano.tensor import slinalg, nlinalg

from matplotlib import pyplot as plt


class GP(GPy.core.Model):
    def __init__(self, X, Y, kernel=None, mean=None, name="GP"):
        super(GP, self).__init__(name=name)
        self.X = X
        self.Y = Y
        if mean is None:
            mean = 0.
        self.mean = mean
        self.kernel = kernel

        self.num_data, self.output_dim, self.input_dim = self.X.shape[0], self.Y.shape[1], self.X.shape[1]        
        self.sigma = GPy.Param('noise', 1., GPy.constraints.Logexp())

        self.link_parameter(self.sigma)
        self.link_parameter(self.kernel)
        
        #Theano init
        if isinstance(self.mean, GPy.Parameterized):
            self.mean.init_theano(self.T_X1, self.T_X2)
            self.link_parameter(self.mean)
        
        self.T_X1 = T.dmatrix('X1')
        self.T_X2 = T.dmatrix('X2')
        self.T_sigma = T.dvector('sigma')

        self.kernel.init_theano(self.T_X1, self.T_X2)
        
        T_K = self.kernel.T_K + T.eye(self.num_data)*(self.T_sigma[0]+1e-6)
        #T_L = slinalg.cholesky(T_K)
        T_Kinv = nlinalg.matrix_inverse(T_K)
        
        Y_mean = (self.Y - self.mean)
        self.T_obj = (- .5*self.num_data*self.output_dim*T.log(2.*np.pi) 
                      - .5*self.output_dim*T.log(nlinalg.det(T_K))
                      #- self.output_dim*nlinalg.trace(T.log(T_L))
                      - .5*nlinalg.trace((T.dot(Y_mean.T, T.dot(T_Kinv, Y_mean))))
                      )
        self.T_grad = theano.grad(self.T_obj, [self.T_sigma] + self.kernel.T_list)
        
        self.f_obj = theano.function([self.T_X1, self.T_X2, self.T_sigma] + kernel.T_list, [self.T_obj, T_Kinv])
        self.f_grad = theano.function([self.T_X1, self.T_X2, self.T_sigma] + kernel.T_list, self.T_grad)
        
    def log_likelihood(self):
        return self._obj
        
    def parameters_changed(self):
        param_list = [self.X, self.X, self.sigma] + self.kernel.parameters
        self._obj, self._Kinv = self.f_obj(*param_list)
        grads = self.f_grad(*param_list)
        self.gradient[:] = np.concatenate(grads)
        
    def predict(self, Xnew):
        Kx = self.kernel.K(Xnew, self.X)
        mu = Kx.dot(self._Kinv.dot(self.Y))
        var = self.kernel.K(Xnew, Xnew) - Kx.dot(self._Kinv).dot(Kx.T) + np.eye(Xnew.shape[0])*self.sigma
            
        return mu, var

    def plot(self, resolution=100):
        if self.input_dim == 1:
            plt.scatter(self.X,self.Y,color='k',marker='x')
            mi, ma = self.X.min(), self.X.max()
            ten_perc = .1*(ma-mi)
            Xpred = np.linspace(mi-ten_perc, ma+ten_perc, resolution)[:, None]
            mu, var = self.predict(Xpred)
            plt.plot(Xpred, mu, color='g', lw=1.5)
            plt.fill_between(Xpred[:, 0], mu[:,0]+2*np.sqrt(np.diagonal(var)), mu[:,0]-2*np.sqrt(np.diagonal(var)), color='k', alpha=.1)
        else:
            raise NotImplementedError("Only one dim plots allowed")
