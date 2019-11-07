import numpy as np

rf = 1.814 #US 10 year tresury

n_obs = 252*5
n_assets = 3

mu = np.array([0.08,0.04,0.02]) #annual return 
ret = np.divide(mu, 252) #daily return

ann_vol = [0.16, 0.08, 0.02] #annual volatility
std = np.diag(np.divide(ann_vol, 252**0.5)) #daily volatility

corr = [1.0, -0.1, 0.0,
        -0.1, 1.0, 0.3,
         0.0, 0.3, 1.0]

corrMat = np.reshape(corr, (3,3)) #correlation matrix
covMat = np.dot(std, np.dot(corrMat, std)) # daily covariance matrix

rebalance_days = 21
n_rebalance = int(252*5/rebalance_days)
n_test_period = 3

def rand_returns(n_assets=n_assets, n_obs=n_obs, print_result=True):    
    #draw samples from the multivariate normal distribution defined 
    #by our target returns and covariance matrix.
    r = np.random.multivariate_normal(ret, covMat, size=n_obs)
    if print_result:
        print('Simulated correlation matix: \n', np.corrcoef(r.T), '\n')
        print('Simulated annual return: \n', np.mean(r, axis=0)*252, '\n')
    return r