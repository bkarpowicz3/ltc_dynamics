import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from os import path
np.random.seed(20211023)

def pendulum_1(curr_state, t, length=1, mass=1, damping_coeff=0, g=9.81):
    """Pendulum with mass, length, and damping if desired

    """
    x = curr_state[0] # current position
    v = curr_state[1] # current velocity
    
    dx_dt = v # update of position is based on current velocity
    dv_dt = -damping_coeff/mass*v - g/length*np.sin(x) # damping + gravitational force
    
    return [dx_dt, dv_dt]

def pendulum_2(curr_state, t, length=0.5, mass=2, damping_coeff=0, g=9.81):
    """Pendulum with mass, length, and damping if desired

    """
    x = curr_state[0] # current position
    v = curr_state[1] # current velocity
    
    dx_dt = v # update of position is based on current velocity
    dv_dt = -damping_coeff/mass*v - g/length*np.sin(x) # damping + gravitational force
    
    return [dx_dt, dv_dt]

def pendulum_3(curr_state, t, length=0.25, mass=4, damping_coeff=0, g=9.81):
    """Pendulum with mass, length, and damping if desired

    """
    x = curr_state[0] # current position
    v = curr_state[1] # current velocity
    
    dx_dt = v # update of position is based on current velocity
    dv_dt = -damping_coeff/mass*v - g/length*np.sin(x) # damping + gravitational force
    
    return [dx_dt, dv_dt]

def generate_trajectory(system, init_state, t_max, t_burn, dt):
    """runs dynamical system forward"""

    # generate time vector
    t = np.arange(0, t_max+t_burn, dt)
    t_start = round(t_burn/dt)
    trajectory = odeint(system, init_state, t)

    return t[t_start:], trajectory[t_start:,:]

def standardize_trajectory(traj, mean_traj=None, stddev_traj=None):

    if mean_traj is None:
        mean_traj = np.mean(traj, axis=0)
    if stddev_traj is None:
        stddev_traj = np.std(traj, axis=0)

    z_traj = np.divide((traj - mean_traj),stddev_traj)
    
    return z_traj

def plot_traj(traj):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2])

    return ax

runpath = '/snel/home/lwimala/tmp/cs7643_project/run_001/'
OVERWRITE = True
save_data = True
n_init_conds = 8
n_trials = 40
dt = 0.010 # seconds
t_max = 40 # seconds
t_burn = 3 # seconds, time to trim from beginning
data_dim = 60
lowd_dim = 3
scale_lowd = 5
chop_len = 100 # bins
chop_olap = 20 # bins
tshift = 0 # bins
t_buffer = 2*tshift*dt # seconds, buffer for tshift
downsample_factor = 5
scale_norm = 0.6
bias = 10 # spk/s
baseline_fr = 10 # spk/s
# compute transformations for mean from dynamical system and set variance to be mean dependent
w_mean = np.random.rand(lowd_dim, data_dim) - 0.5

w_mean_norm = np.linalg.norm(w_mean)*scale_norm
w_mean = np.divide(w_mean, w_mean_norm) # normalize projection matrix

all_lowd_traj = []
for i in range(n_init_conds):
    #init_state = (np.random.rand(1,lowd_dim)-0.5)*20
    low = np.pi/4
    high = np.pi/2
    init_state_1 = np.random.uniform(low=low, high=high, size=(1,2)).tolist()[0]
    init_state_2 = np.random.uniform(low=low, high=high, size=(1,2)).tolist()[0]
    init_state_3 = np.random.uniform(low=low, high=high, size=(1,2)).tolist()[0]
    
    # set velocities to 0
    init_state_1[1] = 0
    init_state_2[1] = 0
    init_state_3[1] = 0
    print(init_state_1)
    t, lowd_dim1 = generate_trajectory(pendulum_1, init_state_1, t_max, t_burn, dt)
    _, lowd_dim2 = generate_trajectory(pendulum_2, init_state_2, t_max, t_burn, dt)
    _, lowd_dim3 = generate_trajectory(pendulum_3, init_state_3, t_max, t_burn, dt)
    
    lowd_traj = np.concatenate( [lowd_dim1[:,0, np.newaxis], lowd_dim2[:,0, np.newaxis], lowd_dim3[:,0,np.newaxis]], axis=1)
    
    # scale_lowd
    lowd_traj *= scale_lowd
    #import pdb; pdb.set_trace()
    traj_inds = np.arange(0, lowd_traj.shape[0], downsample_factor)

    lowd_traj = lowd_traj[traj_inds,:]
    t = t[traj_inds]

    all_lowd_traj.append(lowd_traj)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for lowd_traj in all_lowd_traj:
#     ax.plot(lowd_traj[:,0], lowd_traj[:,1], lowd_traj[:,2])

# compute mean and std from concat traj to standardize system 
concat_traj = np.vstack(all_lowd_traj)
mean_traj = np.mean(concat_traj,axis=0)
stddev_traj = np.std(concat_traj,axis=0)


# geenerate rates from lowd

all_rates = [] # time-varying mean
all_alpha = [] # Gamma dist. params
all_beta = []
all_lowd = []
for i in range(n_init_conds):
    #norm_lowd_traj = minmax_norm_trajectory(lowd_traj)

    #cent_lowd_traj = norm_lowd_traj - np.mean(norm_lowd_traj, axis=0)
    cent_lowd_traj = standardize_trajectory(all_lowd_traj[i], mean_traj=mean_traj, \
                                            stddev_traj=stddev_traj)

    log_mean = np.matmul(cent_lowd_traj, w_mean) 
    fr_mod = np.random.rand(1,data_dim)*bias
    mean = np.exp(log_mean + np.log(fr_mod)) + baseline_fr
    
    # rates 
    rates = mean
    all_rates.append(rates)
    all_lowd.append(cent_lowd_traj)

    if i==0:
        ax = plot_traj(cent_lowd_traj)
    else:
        ax.plot(cent_lowd_traj[:,0], cent_lowd_traj[:,1], cent_lowd_traj[:,2])

concat_rates = np.vstack(all_rates)
concat_lowd = np.vstack(all_lowd)
# create a list that repeats sampling of the concatenated rates
concat_data = [ np.random.poisson(concat_rates*dt) for i in range(n_trials)]   

 # the below allows us to go from 
 # create 3d array shape trials x cond*time x dim
de = np.stack(concat_data)
n_samples, time_cond, dim = de.shape
# get original locations of data for reconstruction
concat_inds = np.arange(de.shape[1])
# reshape indices to be specific to each condition these will be used for reconstruction of chopped data
reshape_inds = np.reshape(concat_inds,[n_init_conds, t.size])

# reshape 3d array to 4d array trials x dim x cond x time
de2 = np.reshape(np.transpose(de, (0,2,1)),[n_samples, dim, n_init_conds, t.size ])

# reshape rates/lowd data to dim x cond x time
truth = np.reshape(concat_rates.T,[dim, n_init_conds, t.size])
truth_lowd = np.reshape(concat_lowd.T, [lowd_dim, n_init_conds, t.size])

if tshift > 0:
    tshift_chop_padding = np.ones((de2.shape[0],de2.shape[1],de2.shape[2], tshift))
    #tshift_chop_padding = np.ones((all_data.shape[0],tshift,all_data.shape[2]))*0.
    de2 = np.concatenate((tshift_chop_padding, de2,tshift_chop_padding), axis=3)
    #all_data = np.hstack([tshift_chop_padding, all_data, tshift_chop_padding])
    

n_chops = int(np.floor(t.size - chop_len) /(chop_len-chop_olap))

all_chops = []
all_truth_chops = []
all_truth_lowd_chops = []
all_chop_inds = []
for i in range(n_chops):
    start_idx = i*chop_len - i*chop_olap + tshift
    end_idx = start_idx + chop_len 
    chop = de2[:,:,:, start_idx-tshift:end_idx+tshift]
    truth_chop = truth[:,:,start_idx:end_idx]
    truth_lowd_chop = truth_lowd[:,:,start_idx:end_idx]
    truth_chops = np.reshape(np.tile(np.transpose(truth_chop, (0,2,1))[:,:,:,np.newaxis], (1,1,1,n_trials)), [dim, chop_len, n_init_conds*n_samples]).T
    truth_lowd_chops = np.reshape(np.tile(np.transpose(truth_lowd_chop, (0,2,1))[:,:,:,np.newaxis], (1,1,1,n_trials)), [lowd_dim, chop_len, n_init_conds*n_samples]).T

    chop_inds = reshape_inds[:,np.arange(start_idx-tshift, end_idx-tshift)]
    chop_inds = np.reshape(np.transpose(np.tile(chop_inds[np.newaxis,:,:], (n_trials,1,1)), (2,1,0)), [ chop_len, n_init_conds*n_samples]).T
    chops = np.reshape(np.transpose(chop,(1,3,2,0)), [dim, chop_len+(2*tshift), n_init_conds*n_samples]).T
    all_chops.append(chops)
    all_truth_chops.append(truth_chops)
    all_truth_lowd_chops.append(truth_lowd_chops)
    all_chop_inds.append(chop_inds)

true_lowd = np.vstack(all_truth_lowd_chops)
true = np.vstack(all_truth_chops)
chops = np.vstack(all_chops)
chop_inds = np.vstack(all_chop_inds)

print('Created %i chops!!' % chops.shape[0])

idx = np.random.permutation(chops.shape[0])
valid_inds = np.sort(idx[::5])
train_inds = np.sort(np.setdiff1d(idx,valid_inds))


datadir = path.join(runpath, 'lfads_input')
modeldir = path.join(runpath, 'lfads_output')

def check_exists(dirpath, overwrite=False):
    if path.exists(dirpath) and path.isdir(dirpath):
        print( 'WARNING: dir exists.' )
        if overwrite:
            import shutil
            print('INFO: Overwriting. Removing %s' % dirpath)
            shutil.rmtree(dirpath)

from collections import namedtuple

ChopParams = namedtuple(
    'ChopParams', [
        'chop_len',
        'chop_olap',
        'dim',
        'nconds',
        't',
        'ntrials_per_cond'
        ])
        

chop_params = ChopParams(
    chop_len=chop_len,
    chop_olap=chop_olap,
    dim=dim,
    nconds=n_init_conds,
    t=t.size,
    ntrials_per_cond=n_trials)

if save_data:
    check_exists(datadir, overwrite=OVERWRITE)
    check_exists(modeldir, overwrite=OVERWRITE)
    os.makedirs(datadir)
    os.makedirs(modeldir)
    filename = 'lfads_simple_pendulum.h5'
    filepath = path.join(datadir, filename)
    with h5py.File(filepath, 'w') as h5f:
        h5f.create_dataset('train_data', data=chops[train_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('train_true', data=true[train_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('train_lowd', data=true_lowd[train_inds,:,:], dtype='float32', compression='gzip')        
        h5f.create_dataset('train_inds', data=train_inds, compression='gzip')
        h5f.create_dataset('train_idx', data=chop_inds[train_inds,:], compression='gzip')
    
        h5f.create_dataset('valid_data', data=chops[valid_inds,:,:], dtype='float32', compression='gzip')    
        h5f.create_dataset('valid_true', data=true[valid_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('valid_lowd', data=true_lowd[valid_inds,:,:], dtype='float32', compression='gzip')        
        h5f.create_dataset('valid_inds', data=valid_inds, compression='gzip')
        h5f.create_dataset('valid_idx', data=chop_inds[valid_inds,:], compression='gzip')
        h5f.create_dataset('chop_params', data=chop_params, compression='gzip')    
        #if hasExtInputs:
        #    h5f.create_dataset('train_ext_input', data=np.transpose(train_ext), compression='gzip')
        #    h5f.create_dataset('valid_ext_input', data=np.transpose(valid_ext), compression='gzip')
        
    print('Sucessfully wrote the data to %s' % filepath)

import pdb; pdb.set_trace()
