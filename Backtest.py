import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from configs import *
import MarkowitzPortfolio
import RiskParityPortfolio
import LSTM_sequence_pred as LS

def get_LSTM_predicted_weights(weights_seqs):
    seqs_train = LS.create_input(weights_seqs[:-1])
    seqs_pred = np.array([weights_seqs[-rebalance_days:-1]])
    LSTM = LS.LSTM_model()
    LSTM.build()
    LSTM.train(seqs_train, split = 0)
    pred_weights = LSTM.predict(seqs_pred)
    return pred_weights

###########################################
### Calculate historical returns
###########################################
returns = rand_returns()
mk_returns = np.empty(0)
rp_returns = np.empty(0)
eq_returns = np.sum(returns/n_assets, axis=1)
for i in range(n_rebalance):
    sub_returns = returns[i*rebalance_days:(i+1)*rebalance_days]
    
    MK = MarkowitzPortfolio.Portfolio()
    MK.get_returns(returns=sub_returns)
    MK.Markowitz(0.5*min(MK.ann_returns)+0.5*max(MK.ann_returns))  
    mk_returns = np.hstack([mk_returns, np.sum(sub_returns*MK.weights, axis=1)])
    mk_weights_true = np.vstack([mk_weights_true, MK.weights])
   
    RP = RiskParityPortfolio.Portfolio()
    RP.get_returns(returns=sub_returns)
    RP.risk_parity()
    rp_returns = np.hstack([rp_returns, np.sum(sub_returns*RP.weights, axis=1)])
    rp_weights_true = np.vstack([rp_weights_true, RP.weights])
    
###########################################
### Calculate historical weights
###########################################
mk_weights_true = np.empty(n_assets)
rp_weights_true = np.empty(n_assets)
for i in range(rebalance_days, n_obs-rebalance_days):
    sub_returns = returns[i-rebalance_days:i]
    
    MK = MarkowitzPortfolio.Portfolio()
    MK.get_returns(returns=sub_returns)
    MK.Markowitz(0.5*min(MK.ann_returns)+0.5*max(MK.ann_returns))  
    mk_weights_true = np.vstack([mk_weights_true, MK.weights])
   
    RP = RiskParityPortfolio.Portfolio()
    RP.get_returns(returns=sub_returns)
    RP.risk_parity()
    rp_weights_true = np.vstack([rp_weights_true, RP.weights])
    
###########################################
### Get predicted weights via LSTM
###########################################
mk_returns_pred = []
rp_returns_pred = [] 
for i in np.arange(n_test_period,0,-1):
    if i == 1:
        sub_returns = returns[-rebalance_days:]
    else:
        sub_returns = returns[-i*rebalance_days:(-i+1)*rebalance_days]

    mk_pred_weights = get_LSTM_predicted_weights(mk_weights_true[:-i*rebalance_days])
    rp_pred_weights = get_LSTM_predicted_weights(rp_weights_true[:-i*rebalance_days])
    mk_returns_pred.extend(np.sum(sub_returns*mk_pred_weights, axis=1).flatten())
    rp_returns_pred.extend(np.sum(sub_returns*rp_pred_weights, axis=1).flatten())

###########################################
### Complare all 
###########################################
n_historical = len(mk_returns)        
n_pred = len(mk_returns_pred)
fig, ax = plt.subplots(figsize=(12,8))
ax.axvline(n_historical-n_pred, color='grey', alpha=0.3)
ax.plot(np.arange(len(eq_returns)),np.cumsum(eq_returns),label='Equal Capital', color='b', alpha=0.4)
ax.plot(np.arange(n_historical), np.cumsum(mk_returns),label='Markowitz',color='r',alpha=0.6)
ax.plot(np.arange(n_historical-n_pred, n_historical), np.sum(mk_returns[:-n_pred])+np.cumsum(mk_returns_pred),'--',color='r', alpha=0.8, label='LSTM_Markowitz')
ax.plot(np.arange(n_historical), np.cumsum(rp_returns),label='Risk Parity',color='g',alpha=0.6)
ax.plot(np.arange(n_historical-n_pred, n_historical), np.sum(rp_returns[:-n_pred])+np.cumsum(rp_returns_pred),'--',color='g', alpha=0.8, label='LSTM_Risk Parity')
ax.plot(np.cumsum(returns[:,0]),alpha=0.2, label='Asset 1')
ax.plot(np.cumsum(returns[:,1]),alpha=0.2, label='Asset 2')
ax.plot(np.cumsum(returns[:,2]),alpha=0.2, label='Asset 3')
ax.tick_params(labelsize=12)
ax.legend()
ax.set_ylabel('Cumulative returns',fontsize=12)
ax.set_title('Strategies Comparison',fontsize=15)

###########################################
### True vs Prediction
###########################################
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.cumsum(mk_returns[-n_pred:]),label='Markowitz',color='r',alpha=0.6)
ax.plot(np.cumsum(mk_returns_pred),'--',color='r', alpha=0.8, label='LSTM_Markowitz')
ax.plot(np.cumsum(rp_returns[-n_pred:]),label='Risk Parity',color='g',alpha=0.6)
ax.plot(np.cumsum(rp_returns_pred),'--',color='g', alpha=0.8, label='LSTM_Risk Parity')
ax.tick_params(labelsize=12)
ax.legend()
ax.set_ylabel('Cumulative returns',fontsize=12)
ax.set_title('Strategies Comparison',fontsize=15)
