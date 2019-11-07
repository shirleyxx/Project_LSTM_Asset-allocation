import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from configs import *
from scipy.optimize import minimize

def risk_objective(w,cov,risk_budget):
    vol = np.sqrt(np.dot(w,np.dot(cov,w)))
    target_risk = np.multiply(vol,risk_budget)
    current_risk = np.multiply(np.dot(cov,w),w)/vol
    return sum(np.square(current_risk - target_risk))
    
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

    def risk_parity(self):      
        w0          = self.n_assets*[1./self.n_assets] #initial weights
        cov         = np.cov(self.returns.T)*252   
        risk_budget = self.n_assets*[1./self.n_assets]
        bnds        = (tuple((0, 1) for _ in range(self.n_assets)))
        cons        = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        w = minimize(risk_objective, w0, args=(cov, risk_budget), method='SLSQP', bounds=bnds, constraints=cons)['x']
        self.weights = w
        self.risk_contrib = np.multiply(np.dot(cov,w),w)/np.sqrt(np.dot(w,np.dot(cov,w)))

if __name__ == "__main__":
    p = Portfolio()
    p.get_returns(simulate=True)
    p.risk_parity()
    print('Risk parity portfolio weights:\n',p.weights)
    print('Risk parity portfolio risk contribution:\n',p.risk_contrib)
  