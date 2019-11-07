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
### Calculate historical weights, returns
###########################################
returns = rand_returns()
mk_returns = np.empty(0)
rp_returns = np.empty(0)
eq_returns = np.sum(returns/n_assets, axis=1)

for i in range(n_rebalance-n_test_period):
    sub_returns = returns[i*rebalance_days:(i+1)*rebalance_days]
    
    MK = MarkowitzPortfolio.Portfolio()
    MK.get_returns(returns=sub_returns)
    MK.Markowitz(min(MK.ann_returns))  
    mk_returns = np.hstack([mk_returns, np.sum(sub_returns*MK.weights, axis=1)])
 
    RP = RiskParityPortfolio.Portfolio()
    RP.get_returns(returns=sub_returns)
    RP.risk_parity()
    rp_returns = np.hstack([rp_returns, np.sum(sub_returns*RP.weights, axis=1)])
     
mk_weights_seqs = np.empty(n_assets)
rp_weights_seqs = np.empty(n_assets)
for i in range(rebalance_days, n_obs-n_test_period*rebalance_days):
    sub_returns = returns[i-rebalance_days:i]
    
    MK = MarkowitzPortfolio.Portfolio()
    MK.get_returns(returns=sub_returns)
    MK.Markowitz(min(MK.ann_returns))  
    mk_weights_seqs = np.vstack([mk_weights_seqs, MK.weights])
    
    RP = RiskParityPortfolio.Portfolio()
    RP.get_returns(returns=sub_returns)
    RP.risk_parity()
    rp_weights_seqs = np.vstack([rp_weights_seqs, RP.weights])
     
mk_pred_weights = get_LSTM_predicted_weights(mk_weights_seqs)
rp_pred_weights = get_LSTM_predicted_weights(rp_weights_seqs)
sub_returns = returns[(n_rebalance-n_test_period)*rebalance_days: n_rebalance*rebalance_days]
mk_returns_pred = np.sum(sub_returns*mk_pred_weights, axis=1)
rp_returns_pred = np.sum(sub_returns*rp_pred_weights, axis=1)

###########################################
### Plot
###########################################
n_historical = len(mk_returns)        
n_pred = len(mk_returns_pred)
fig, ax = plt.subplots()
ax.axvline(n_historical-1, color='grey', alpha=0.3)
ax.plot(np.arange(len(eq_returns)),np.cumsum(eq_returns),label='Equal Capital', color='b', alpha=0.4)
ax.plot(np.arange(n_historical), np.cumsum(mk_returns),label='Markowitz Min Variance',color='r',alpha=0.6)
ax.plot(np.arange(n_historical, n_historical+n_pred), np.sum(mk_returns)+np.cumsum(mk_returns_pred),'--',color='r', alpha=0.6, label='LSTM_Markowitz')
ax.plot(np.arange(n_historical), np.cumsum(rp_returns),label='Risk Parity',color='g',alpha=0.6)
ax.plot(np.arange(n_historical, n_historical+n_pred), np.sum(rp_returns)+np.cumsum(rp_returns_pred),'--',color='g', alpha=0.6, label='LSTM_Risk Parity')
ax.tick_params(labelsize=12)
ax.legend()
ax.set_ylabel('Cumulative returns',fontsize=12)
ax.set_title('Strategies Comparison',fontsize=15)

