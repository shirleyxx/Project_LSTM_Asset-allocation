import numpy as np
import cvxopt as opt
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from configs import *

# Turn off progress printing 
opt.solvers.options['show_progress'] = False

class Portfolio():
    def __init__(self, n_assets=n_assets, n_obs=n_obs):
        self.n_assets = n_assets
        self.n_obs = n_obs
        
    def get_returns(self, simulate=False, returns=None):
        if simulate:
            self.returns = rand_returns(self.n_assets, self.n_obs)
        else:
            self.returns = returns
        self.ann_returns = np.mean(self.returns, axis=0)*252
    
    def Markowitz(self, r_threshold):
        n = self.n_assets
       
        # Convert to cvxopt matrices
        Cov = opt.matrix(np.cov(self.returns.T)*252)
        r = opt.matrix(np.mean(self.returns, axis=0)*252)
        
        # Create constraint matrices
        G = -opt.matrix(np.vstack([np.eye(n), self.ann_returns]))   # negative n x n identity matrix
        h = opt.matrix(np.vstack([np.zeros((n,1) ),[-r_threshold]]))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
    
        # Calculate optimal weights
        w = opt.solvers.qp(Cov, 0*r, G, h, A, b)['x']   
        
        # Calculate optimal return and risk
        optimal_weights = list(w)
        optimal_return = opt.blas.dot(r, w)
        optimal_vol = np.sqrt(opt.blas.dot(w, Cov*w))
        self.weights = optimal_weights
        return optimal_weights, optimal_return, optimal_vol
     
    def efficient_frontier(self):
        self.optimal_returns = []
        self.optimal_vols = []
        return_thresholds = np.linspace(min(self.ann_returns),max(self.ann_returns),150)
        for r_th in return_thresholds:
            w, r, vol = self.Markowitz(r_th)
            self.optimal_returns.append(r)
            self.optimal_vols.append(vol)
    
    def plot_efficient_frontier(self):
        plt.figure(figsize=(8,6))
        plt.grid(True)
        ax = plt.gca()
        ax.plot(self.optimal_vols, self.optimal_returns, 'o', markersize=5, alpha=0.7)
       # ax.plot(self.optimal_vols, self.optimal_returns, 'lightseagreen')
        ax.set_xlabel('ann vol', fontsize=12)
        ax.set_ylabel('ann return',fontsize=12)
        ax.set_title('Efficient Frontier',fontsize=15)
        ax.tick_params(labelsize=12)
        
if __name__ == "__main__":
    p = Portfolio()
    p.get_returns(simulate=True)
    p.Markowitz(0.0)
    p.efficient_frontier()
    p.plot_efficient_frontier()