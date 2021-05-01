#%% 
from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=0)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from os import path
import numpy as np
import h5py
import glob
import pandas as pd
# from lfads_tf2.subclasses.ltc.models import LTC_LFADS
from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults

#%%
cfg = get_cfg_defaults()
cfg.TRAIN.DATA.DIR = '/snel/home/brianna/projects/deep_learning_project/lfads_input_nonlinear_coupling/'
cfg.TRAIN.MODEL_DIR = '/snel/home/brianna/projects/deep_learning_project/lfads_output_nonlinear_coupling/'
cfg.TRAIN.DATA.PREFIX = 'lfads'

spikes  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='data', merge_tv=True)[0]
true_rates  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='true', merge_tv=True)[0]
lowd  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='lowd', merge_tv=True)[0]
idx  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='idx', merge_tv=True)[0]

with h5py.File(glob.glob(path.join(cfg.TRAIN.DATA.DIR, cfg.TRAIN.DATA.PREFIX + '*'))[0], 'r') as h5file:
    chop_len, chop_olap, dim, nconds, t, ntrials_per_cond = h5file['chop_params'][()]

lfads_output = load_posterior_averages(cfg.TRAIN.MODEL_DIR, merge_tv=True)
lfads_rates = lfads_output.rates
lfads_factors = lfads_output.factors
#%% 
def merge_chops( chop_data, chop_len, chop_olap, dim=None ):

    w_left = np.arange(chop_olap)/chop_olap
    w_right = np.flip(w_left)

    wt = np.ones((chop_len))
    wt[:chop_olap] = w_left
    wt[-chop_olap:] = w_right

    if dim is None:
        dim = chop_data.shape[2]
    full_data =np.zeros((ntrials_per_cond,nconds, t, dim))
    
    for i in range(idx.shape[0]):
        trial_idx = np.mod(i,ntrials_per_cond)
        cond_idx = np.floor(idx[i,-1]/t).astype(np.int32)
        time_inds = np.mod(idx[i,:], t).astype(np.int32)

        # NOTE: Leftmost chop not handled properly to avoid applying weights to left edge 
        full_data[trial_idx,cond_idx,time_inds,:] += \
                                                     np.multiply(chop_data[i,:,:], \
                                                                 wt[:,np.newaxis])

    return full_data

lfads_rates_full = merge_chops(lfads_rates, chop_len, chop_olap)
lfads_factors_full = merge_chops(lfads_factors, chop_len, chop_olap)
spikes_full = merge_chops(spikes, chop_len, chop_olap)
true_rates_full = merge_chops(true_rates, chop_len, chop_olap)
lowd_full = merge_chops(lowd, chop_len, chop_olap)

nchops = int(np.floor(t - chop_len) /(chop_len-chop_olap))
end_idx = t - (chop_len + (chop_len-chop_olap)*(nchops-1))

lfads_rates_full = lfads_rates_full[:,:,chop_olap:-(end_idx+chop_olap)]
lfads_factors_full = lfads_factors_full[:,:,chop_olap:-(end_idx+chop_olap)]
spikes_full = spikes_full[:,:,chop_olap:-(end_idx+chop_olap)]
true_rates_full = true_rates_full[:,:,chop_olap:-(end_idx+chop_olap)]
lowd_full = lowd_full[:,:,chop_olap:-(end_idx+chop_olap)]

#%%
lowd_flat = lowd_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], lowd_full.shape[3])
lfads_rates_flat = lfads_rates_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
lfads_factors_flat = lfads_factors_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], 12)
spikes_flat = spikes_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
true_rates_flat = true_rates_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)

#%% Quantify reconstruction of firing rates 
def R2(x, y):
    '''
    x: the true/known values 
    y: the predicted values 
    outputs R^2 value
    '''
    return 1 - (sum((x - y)**2) / sum((x - np.mean(x))**2))


lfads_r2 = R2(true_rates_flat, lfads_rates_flat)
ltc_r2 = [1, 0.9, 0.9]
vafs = [np.mean(lfads_r2), np.mean(ltc_r2)]
yerr = [np.std(lfads_r2), np.std(ltc_r2)]

fig, ax = plt.subplots()
labels = ['LFADS', 'LTC-LFADS']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
rects1 = ax.bar(x, vafs, width, yerr=yerr)
ax.set_ylabel('VAF')
ax.set_title('Firing Rate Reconstruction Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim([-0.5,1.5])

#%% Plot time series firing rates 
# neurons = sorted(np.random.randint(0, lfads_rates_flat.shape[-1], 5))
neurons = [2, 6, 21, 29, 40]
fig, ax = plt.subplots(5,1, figsize=(6, 10))
for ii, n in enumerate(neurons):
    ax[ii].plot(true_rates_flat[0:500, n], color='k', label='True')
    ax[ii].plot(lfads_rates_flat[0:500, n]/0.01, color='r', label='LFADS')
    ax[ii].set_ylabel('Neuron ' + str(n))
ax[0].legend()
ax[-1].set_xlabel('Time (s)')
fig.suptitle('Estimation of LFADS Rates')

# %% Estimate of actual dynamics 
from sklearn.linear_model import LinearRegression

x = lfads_factors_flat
y = lowd_flat
# fits intercept
lr = LinearRegression().fit(x,y) # object

# get predictions
lowd_hat = lr.predict(x)
print(R2(y, lowd_hat))

n_dynamics = lowd_hat.shape[-1]
fig, ax = plt.subplots(n_dynamics,1)
for ii in range(n_dynamics):
    ax[ii].plot(lowd_flat[0:500, ii], color='k', label='True')
    ax[ii].plot(lowd_hat[0:500, ii], color='r', label='LFADS')
    ax[ii].set_ylabel('Dynamics Dim ' + str(ii))
ax[0].legend()
fig.suptitle('Estimation of Composite Dynamics')

# %% Training Curves 

lfads_train_data = os.path.join(cfg.TRAIN.MODEL_DIR, 'train_data.csv')
lfads_df = pd.read_csv(lfads_train_data)

epochs = lfads_df.epoch.values 
train_loss = lfads_df.loss.values
valid_loss = lfads_df.val_loss.values
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, valid_loss, label='Valid')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('LFADS Training')
plt.legend()

# %%
