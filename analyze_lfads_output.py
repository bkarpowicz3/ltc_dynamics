#%% 
import os
import pandas as pd
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
from generate_spiking_data import LTCDataset
from lfads_tf2.utils import load_posterior_averages

#%% 
input_data_path = '/snel/home/brianna/projects/deep_learning_project/rds_nonlinearOscillator.pkl'
lfads_best_model_path = '/snel/home/brianna/projects/deep_learning_project/210426_PBT/pbt_run_7/best_model/'
ltc_best_model_path = ''

# load RDS 
with open(input_data_path, 'rb') as f: 
    ds = pickle.load(f)

# load LFADS output
sampling_output = load_posterior_averages(lfads_best_model_path, merge_tv=True)
lfads_rates = sampling_output.rates
lfads_factors = sampling_output.factors

train_inds = ds.elf.train_inds
valid_inds = ds.elf.valid_inds 

rates = np.zeros((len(train_inds) + len(valid_inds), lfads_rates.shape[1], lfads_rates.shape[2]))
rates[train_inds, :, :] = lfads_rates[train_inds, :, :]
rates[valid_inds, :, :] = lfads_rates[valid_inds, :, :] 
lfads_rates = np.vstack(rates)

factors = np.zeros((len(train_inds) + len(valid_inds), lfads_factors.shape[1], lfads_factors.shape[2]))
factors[train_inds, :, :] = lfads_factors[train_inds, :, :]
factors[valid_inds, :, :] = lfads_factors[valid_inds, :, :] 
lfads_factors = np.vstack(factors)

# does not work due to the way data generation has changed
# ds.merge_data_to_df(lfads_rates, new_fieldname='lfads_rates', smooth_overlaps=True)
# ds.merge_data_to_df(lfads_factors, new_fieldname='lfads_factors', smooth_overlaps=True)

# ltc_sampling_output = load_posterior_averages(ltc_best_model_path, merge_tv=True)
# ltc_rates = ltc_sampling_output.rates
# ltc_factors = ltc_sampling_output.factors

# ds.merge_data_to_df(ltc_rates, new_fieldname='ltc_rates', smooth_overlaps=True)
# ds.merge_data_to_df(ltc_factors, new_fieldname='ltc_factors', smooth_overlaps=True)

#%% Quantify reconstruction of firing rates 

def R2(x, y):
    '''
    x: the true/known values 
    y: the predicted values 
    outputs R^2 value
    '''
    return 1 - (sum((x - y)**2) / sum((x - np.mean(x))**2))

true_rates = ds.data.true_rates.values
# lfads_rates = ds.data.lfads_rates.values
# ltc_rates = ds.data.ltc_rates.values

lfads_r2 = R2(true_rates, lfads_rates)
# ltc_r2 = R2(true_rates, ltc_rates)
ltc_r2 = 1
vafs = [np.mean(lfads_r2), np.mean(ltc_r2)]

fig, ax = plt.subplots()
labels = ['LFADS', 'LTC-LFADS']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
rects1 = ax.bar(x, vafs, width)
ax.set_ylabel('VAF')
ax.set_title('Firing Rate Reconstruction Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlim([-0.5,1.5])

# %% Time series overlays 

true_lowd = ds.data.lowD_components.values
# lfads_factors = ds.data.lfads_factors.values
# ltc_factors = ds.data.ltc_factors.values 

fig, ax = plt.subplots(3,1)
ax[0].plot(true_lowd[0:500, 0], color='k', label='True')
ax[0].plot(lfads_factors[0:500, 0], color='r', label='LFADS')
ax[1].plot(true_lowd[0:500, 1], color='k')
ax[1].plot(lfads_factors[0:500, 1], color='r')
ax[2].plot(true_lowd[0:500, 2], color='k')
ax[2].plot(lfads_factors[0:500, 2], color='r')

# fig, ax = plt.subplots(3,1)
# ax[0].plot(true_lowd[0:500, 0], color='k', label='True')
# ax[0].plot(ltc_factors[0:500, 0], color='b', label='LTC')
# ax[1].plot(true_lowd[0:500, 1], color='k')
# ax[1].plot(ltc_factors[0:500, 1], color='b')
# ax[2].plot(true_lowd[0:500, 2], color='k')
# ax[2].plot(ltc_factors[0:500, 2], color='b')

# %% Model training curves 
lfads_train_data = os.path.join(lfads_best_model_path, 'train_data.csv')
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

# ltc_train_data = os.path.join(ltc_best_model_path, 'train_data.csv')
# ltc_df = pd.read_csv(ltc_train_data)

# epochs = ltc_df.epoch.values 
# ltc_train_loss = ltc_df.loss.values
# ltc_valid_loss = ltc_df.val_loss.values
# plt.figure()
# plt.plot(epochs, ltc_train_loss, label='Train')
# plt.plot(epochs, ltc_valid_loss, label='Valid')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.title('LTC-LFADS Training')
# plt.legend()
# %% Training curves comparison 

# plt.figure()
# plt.plot(epochs, train_loss, label='LFADS')
# plt.plot(epochs, ltc_train_loss, label='LTC-LFADS')
# plt.legend()
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.title('Train Loss Comparison')

# plt.figure()
# plt.plot(epochs, valid_loss, label='LFADS')
# plt.plot(epochs, ltc_valid_loss, label='LTC-LFADS')
# plt.legend()
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.title('Valid Loss Comparison')
