# A slightly modified version of this: https://github.com/AmazaspShumik/Bayesian-Regression-Methods
# the modifications include:
# switching default iteration method to EM (more stable on lower rank problems)
# changing the API to match the scikit-learn standard where fit takes X,Y instead of init
# added ridge lamda to stabilize

# no license specified.

import numpy as np

class BayesianRegression():
    '''
    Bayesian Regression with type II maximum likelihood for determining point estimates
    for precision variables alpha and beta, where alpha is precision of prior of weights
    and beta is precision of likelihood
    
    Parameters:
    -----------
    
    X: numpy array of size 'n x m'
       Matrix of explanatory variables (should not include bias term)
       
    Y: numpy arra of size 'n x 1'
       Vector of dependent variables (we assume)
       
    thresh: float
       Threshold for convergence for alpha (precision of prior)
       
    '''
    
    def __init__(self,bias_term=True,thresh=1e-5,lam=1e-5):

        self.bias_term         = bias_term        
        
        # precision parameters, they are calculated during evidence approximation
        self.alpha             = None 
        self.beta              = None
        
        # mean and precision of posterior distribution of weights
        self.w_mu              = None
        self.w_precision       = None   # when value is assigned it is m x m matrix
        self.D                 = None   # covariance

        self.thresh            = thresh
        self.lam               = lam
            
    def fit(self,X,Y,evidence_approx_method="EM",max_iter=100):
        '''
        Fits Bayesian linear regression, returns posterior mean and preision 
        of parameters
        
        Parameters:
        -----------
        
        max_iter: int
            Number of maximum iterations
            
        evidence_approx_method: str
            Method for approximating evidence
        
        Returns:
        --------
        
        parameters: dictionary 
                    - parameters['bias_term']  - coefficient for vector of constants
                    - parameters['weights']    - mean of posterior of weights
                    - parameters['precision']  - inverse of covariance for posterior 
                                                 of weights
        '''
        parameters = {}
        
        # center input data for simplicity of further computations
        self.mu_X              =  np.mean(X,axis = 0)
        self.X                 =  X - np.outer(self.mu_X, np.ones(X.shape[0])).T
        self.mu_Y              =  np.mean(Y)
        self.Y                 =  Y - np.mean(Y)
        if self.bias_term is True:
            self.bias_term     =  self.mu_Y
        
        # to speed all further computations save svd decomposition and reuse it later
        self.u,self.d,self.v   =  np.linalg.svd(self.X, full_matrices = False)
        
        # ridge penalty to avoid computational issues:
        self.d += self.lam     
        
        # use type II maximum likelihood to find hyperparameters alpha and beta
        self._evidence_approx(max_iter = max_iter, method = evidence_approx_method)

        # find parameters of posterior distribution after last ipdate of alpha & beta
        self.w_mu, self.w_precision =  self._posterior_params(self.alpha,self.beta)
        d                           =  1/(self.beta*self.d**2 + self.alpha)
        self.D                      =  np.dot( np.dot( self.v.T , np.diag(d) ) , self.v)
        
        
        if self.bias_term is not False:
            parameters["bias_term"] = self.mu_Y
        parameters["weights"]       = self.w_mu
        parameters["precision"]     = self.w_precision
        
        return parameters
            

    def predict_dist(self,x):
        '''
        Calculates  mean and variance of predictive distribution at each data 
        point of test set.
        
        Parameters:
        -----------
        
        x: numpy array of size 'unknown x m'
            Set of features for which corresponding responses should be predicted

        
        Returns:
        ---------
        
        :list of two numpy arrays (each has size 'unknown x 1')
            Parameters of univariate gaussian distribution [mean and variance] 
        
        '''
        x            =  x - self.mu_X
        mu_pred      =  np.dot(x,self.w_mu) + self.mu_Y
        var_pred     =  1/self.beta + np.sum( np.dot( x, self.D )* x, axis = 1 )
        return [mu_pred,var_pred]
        
    
    def predict(self,x,random_draw=False):
        '''
        Calculates mean of predictive distribution at each data point of test set
        
        Parameters:
        ----------
                
        x: numpy array of size 'unknown x m'
            Set of features for which corresponding responses should be predicted
            
        Returns:
        --------
        
        mu_pred: numpy array of size 'unknown x 1'
                  Mean of predictive distribution
            
        '''
        x           = x - self.mu_X
        if random_draw:
            w_mu_draw   = np.random.multivariate_normal(self.w_mu,self.D)
            mu_pred     = np.dot(x,w_mu_draw) + self.mu_Y
        else:
            mu_pred     = np.dot(x,self.w_mu) + self.mu_Y
        return mu_pred  
        
                
    def _evidence_approx(self,max_iter=100,method="EM"):
        '''
        Performs evidence approximation , finds precision  parameters that maximize 
        type II likelihood. There are two different fitting algorithms, namely EM
        and fixed-point algorithm, empirical evidence shows that fixed-point algorithm
        is faster than EM
        
        Parameters:
        -----------
        
        max_iter: int
              Maximum number of iterations
        
        method: str
              Can have only two values : "EM" or "fixed-point"
        
        '''
        
        # number of observations and number of paramters in model
        n,m         = np.shape(self.X)
        
        # initial values of alpha and beta 
        alpha, beta = np.random.random(2)
        
        # store log likelihood at each iteration
        log_likes   = []
        dsq         =  self.d**2

        
        for i in range(max_iter):
            
            # find mean for posterior of w ( for EM this is E-step)
            p1_mu   =  np.dot(self.v.T, np.diag(self.d/(dsq+alpha/beta)))
            p2_mu   =  np.dot(self.u.T, self.Y)
            mu      =  np.dot(p1_mu,p2_mu)
            
            # precompute errors, since both methods use it in estimation
            error   = self.Y - np.dot(self.X,mu)
            sqdErr  = np.dot(error,error)
            
            if method == "fixed-point":
     
                # update gamma
                gamma      =  np.sum(beta*dsq/(beta*dsq + alpha))
               
                # use updated mu and gamma parameters to update alpha and beta
                alpha      =  gamma/np.dot(mu,mu)
                beta       =  (n - gamma)/sqdErr
               
            elif method == "EM":
                
                # M-step, update parameters alpha and beta
                alpha      = m / (np.dot(mu,mu) + np.sum(1/(beta*dsq+alpha)))
                beta       = n / ( sqdErr + np.sum(dsq/(beta*dsq + alpha)))
            
            else:
                raise ValueError("Only 'EM' and 'fixed-point' algorithms are implemented ")
            
            # after alpha & beta are updated last time we should also update mu
            p1_mu   =  np.dot(self.v.T, np.diag(self.d/(dsq+alpha/beta)))
            p2_mu   =  np.dot(self.u.T,self.Y)
            mu      =  np.dot(p1_mu,p2_mu)
            
            # calculate log likelihood p(Y | X, alpha, beta) (constants are not included)
            normaliser =  m/2*np.log(alpha) + n/2*np.log(beta) - 1/2*np.sum(np.log(beta*dsq+alpha))
            log_like   =  normaliser - alpha/2*np.dot(mu,mu) - beta/2*sqdErr           
            log_likes.append(log_like)
            
            # if change in log-likelihood is smaller than threshold stop iterations
            if i >=1:
                if log_likes[-1] - log_likes[-2] < self.thresh:
                    break
        
        # write optimal alpha and beta to instance variables
        self.alpha = alpha
        self.beta  = beta  
        
              
    def _posterior_params(self,alpha,beta):
        '''
        Calculates parameters of posterior distribution of weights.
        Uses economy svd for fast calculaltions.
        
        # Small Theory note:
        ---------------------
        Multiplying likelihood of data on prior of weights we obtain distribution 
        proportional to posterior of weights. By completing square in exponent it 
        is easy to prove that posterior distribution is Gaussian.
        
        Parameters:
        -----------
        
        alpha: float
            Precision parameter for prior distribution of weights
            
        beta: float
            Precision parameter for noise distribution
            
        Returns:
        --------
        
        : list of two numpy arrays
           First element of list is mean and second is precision of multivariate 
           Gaussian distribution
           
        '''
        precision             = beta*np.dot(self.X.T,self.X)+alpha
        self.diag             = self.d/(self.d**2 + alpha/beta)
        p1                    = np.dot(self.v.T,np.diag(self.diag))
        p2                    = np.dot(self.u.T,self.Y)
        w_mu                  = np.dot(p1,p2)
        return [w_mu,precision]