#%% 
from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=0)
import matplotlib
matplotlib.use('Agg')
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
cfg.TRAIN.DATA.PREFIX = 'lfads'

experiments = ['nonlinear_coupling', 'linear_coupling', 'concat3']
data = {}

for ex in experiments: 
    cfg.TRAIN.DATA.DIR = '/snel/home/brianna/projects/deep_learning_project/lfads_input_' + ex + '/'
    if "coupling" in ex:
        cfg.TRAIN.MODEL_DIR = '/snel/home/brianna/projects/deep_learning_project/lfads_output_' + ex + '_updatedhps/'
    else:
        cfg.TRAIN.MODEL_DIR = '/snel/home/brianna/projects/deep_learning_project/lfads_output_' + ex + '/'

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

    data[ex] = {'spikes': spikes,
                'rates': true_rates,
                'lowd': lowd, 
                'idx': idx,
                'lfads_rates': lfads_rates,
                'lfads_factors': lfads_factors}

    # Training Curves 

    lfads_train_data = path.join(cfg.TRAIN.MODEL_DIR, 'train_data.csv')
    lfads_df = pd.read_csv(lfads_train_data)

    epochs = lfads_df.epoch.values 
    train_loss = lfads_df.loss.values
    valid_loss = lfads_df.val_loss.values
    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, valid_loss, label='Valid')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('LFADS Training - ' + ex)
    plt.legend()
    plt.savefig('learning_curve_'+ex+'.png')
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

for ex in experiments:

    d = data[ex]
    lfads_rates = d['lfads_rates']
    lfads_factors = d['lfads_factors']
    spikes = d['spikes']
    true_rates = d['rates']
    lowd = d['lowd']

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

    lowd_flat = lowd_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], lowd_full.shape[3])
    lfads_rates_flat = lfads_rates_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
    lfads_factors_flat = lfads_factors_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], 12)
    spikes_flat = spikes_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
    true_rates_flat = true_rates_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)

    data[ex]['lfads_rates_full'] = lfads_rates_full
    data[ex]['lfads_factors_full'] = lfads_factors_full
    data[ex]['spikes_full'] = spikes_full
    data[ex]['rates_full'] = true_rates_full
    data[ex]['lowd_full'] = lowd_full
    data[ex]['lowd_flat'] = lowd_flat
    data[ex]['lfads_rates_flat'] = lfads_rates_flat
    data[ex]['lfads_factors_flat'] = lfads_factors_flat
    data[ex]['spikes_flat'] = spikes_flat
    data[ex]['rates_flat'] = true_rates_flat

#%% Quantify reconstruction of firing rates 
def R2(x, y):
    '''
    x: the true/known values 
    y: the predicted values 
    outputs R^2 value
    '''
    return 1 - (sum((x - y)**2) / sum((x - np.mean(x))**2))

lfads_means = []
ltc_means = []

fig, ax = plt.subplots()
labels = experiments[:-1]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

for ex in experiments: 
    if 'concat' not in ex:
        true_rates_flat = data[ex]['rates_flat']
        lfads_rates_flat = data[ex]['lfads_rates_flat']

        lfads_r2 = R2(true_rates_flat, lfads_rates_flat)
        ltc_r2 = [1, 0.9]
        lfads_means.append(np.mean(lfads_r2))
        ltc_means.append(np.mean(ltc_r2))
    # vafs = [np.mean(lfads_r2), np.mean(ltc_r2)]
    # yerr = [np.std(lfads_r2), np.std(ltc_r2)]

rects1 = ax.bar(x - width/2, lfads_means, width, label='LFADS')
rects2 = ax.bar(x + width/2, ltc_means, width, label='LTC-LFADS')
ax.set_ylabel('Mean VAF')
ax.set_title('Firing Rate Reconstruction Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig('firing_rates_vaf.png')

#%% Plot time series firing rates 
# neurons = sorted(np.random.randint(0, lfads_rates_flat.shape[-1], 5))
neurons = [2, 6, 21, 29, 40]
fig, ax = plt.subplots(5,3, figsize=(18, 10), sharex=True)
for jj, ex in enumerate(experiments):
    true_rates_flat = data[ex]['rates_flat']
    lfads_rates_flat = data[ex]['lfads_rates_flat']

    for ii, n in enumerate(neurons):
        ax[ii][jj].plot(true_rates_flat[0:500, n], color='k', label='True')
        ax[ii][jj].plot(lfads_rates_flat[0:500, n]/0.01, color='r', label='LFADS')
        ax[ii][jj].set_ylabel('Neuron ' + str(n))
        ax[0][jj].set_title(ex)

ax[0][0].legend()
ax[-1][-1].set_xlabel('Time (s)')
fig.suptitle('Estimation of LFADS Rates')
plt.savefig('firing_rates_time.png')

# %% Estimate of actual dynamics 
from sklearn.linear_model import LinearRegression

for ex in experiments: 

    lfads_factors_flat = data[ex]['lfads_factors_flat']
    lowd_flat = data[ex]['lowd_flat']

    x = lfads_factors_flat
    y = lowd_flat
    # fits intercept
    lr = LinearRegression().fit(x,y) # object

    # get predictions
    lowd_hat = lr.predict(x)
    r2 = R2(y, lowd_hat)
    print(r2)

    n_dynamics = lowd_hat.shape[-1]
    height = 10 if ex == 'concat3' else 5
    fig, ax = plt.subplots(n_dynamics,1, figsize=(6, height))
    for ii in range(n_dynamics):
        ax[ii].plot(lowd_flat[0:500, ii], color='k', label='True')
        ax[ii].plot(lowd_hat[0:500, ii], color='r', label='LFADS')
        ax[ii].set_ylabel('Dynamics Dim ' + str(ii))
    ax[0].legend()
    fig.suptitle('Estimation of Composite Dynamics - ' + ex + '\n Mean VAF = ' + str(np.mean(r2)))
    plt.savefig('dynamics_'+ex+'.png')


# %% Prediction of the components of dynamics

lowd_flat = data['concat3']['lowd_flat']

for ex in experiments: 

    lfads_factors_flat = data[ex]['lfads_factors_flat']
    
    x = lfads_factors_flat
    y = lowd_flat
    # fits intercept
    lr = LinearRegression().fit(x,y) # object

    # get predictions
    lowd_hat = lr.predict(x)
    r2 = R2(y, lowd_hat)
    print(r2)

    n_dynamics = lowd_hat.shape[-1]
    fig, ax = plt.subplots(n_dynamics,1, figsize=(6, 10))
    for ii in range(n_dynamics):
        ax[ii].plot(lowd_flat[0:500, ii], color='k', label='True')
        ax[ii].plot(lowd_hat[0:500, ii], color='r', label='LFADS')
        ax[ii].set_ylabel('Dynamics Dim ' + str(ii))
    ax[0].legend()
    fig.suptitle('Estimation of Component Dynamics - ' + ex + '\n Mean VAF = ' + str(np.mean(r2)))
    plt.savefig('dynamics_components'+ex+'.png')

# %%
