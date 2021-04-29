import numpy as np 
import matplotlib.pyplot as plt 
from rds.structures import Dataset

class LTCDataset(Dataset): 
    def generate_spikes(self):
        ### define the parameters of the simulation
        trial_length = 1 # length of each trial, s 
        self.dt = 0.02 # timestep, s
        dt = self.dt
        self.num_trials = 50 # number of trials to simulate 
        num_trials = self.num_trials
        num_ics = 4 # number of initial conditions to sample 
        ics = [(1,0), (0, -1), (-1, 0), (0, 1)] # the set of initial conditions (x1, x2)

        period = 1 # oscillator period 
        amplitudes = range(1,9) # oscillator amplitude 

        num_neurons = 100 # dimension of the high D space

        mu = 0 # mean of noise 
        sigma = 1 # standard deviation of noise

        length = trial_length * num_trials

        ### define function to create oscillator 
        def oscillator(length, dt, period=1, amplitude=1, ic=1):
            # state vector x is [position, velocity]
            num_timesteps = int(length / dt) # number of time steps
            x = np.zeros((num_timesteps, 1)) # initialize a vector to hold the state

            t = np.arange(0, num_timesteps) * dt # time vector 
            phi = np.pi/2 # phase shift, set to multiples of pi/2
            # model as an oscillator 
            # NOTE: here we use a sine wave for both, and the IC is converted into a phase shift
            # the phase shift preserves the intended dynamics (since sine and cosine are related by pi/2)
            x[:,0] = np.sin(2 * np.pi * t / period + ic*phi) * amplitude 

            return x 
        
        def create_highd_space(lowd, num_neurons):

            # don't care what the high-D space is, so use random
            # balance it by subtracting 0.5
            highd_proj = np.random.rand(lowd.shape[1], num_neurons)-0.5

            # each neuron can also have a different mean "rate"
            # assign these randomly
            log_means = (np.random.rand(num_neurons) - 0.5) * 0.01

            return highd_proj, log_means
        
        def project_to_highd(lowd, highd_proj, log_means):
            # project the low-D data out to high-D space
            z_true = np.dot(lowd, highd_proj) + log_means
            # exponential nonlinearity to convert these to firing rates 
            z_true_rates = (np.exp(z_true) + 3) * 4

            spikes = np.random.poisson(z_true_rates*dt)

            return spikes, z_true_rates

        xs = []
        for a in amplitudes: 
            ### generate oscillators with three different sets of parameters 
            x1 = oscillator(length, dt, period=1, amplitude=a*0.5, ic=1)
            x2 = oscillator(length, dt, period=1.5, amplitude=a*0.5, ic=0)
            x3 = oscillator(length, dt, period=0.5, amplitude=a*0.5, ic=0.5)

            ### visualize components of compound system 
            t = np.linspace(0, length, int(length / dt))
            fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
            ax[0].plot(t[0:1000], x1[0:1000])
            ax[1].plot(t[0:1000], x2[0:1000])
            ax[2].plot(t[0:1000], x3[0:1000])
            plt.show(block=False)

            x_comps = [x1, x2, x3]

            ### perform a nonlinear combination of the three above components and plot 
            x = (x1 + x2)**2 + x3**2
            # center around 0 
            x = (x - np.mean(x)) / np.std(x)

            fig, ax = plt.subplots(4,1)
            ax[0].plot(t[0:500], x1[0:500], color='b')
            ax[0].set_ylabel('X1')
            ax[1].plot(t[0:500], x2[0:500], color='r')
            ax[1].set_ylabel('X2')
            ax[2].plot(t[0:500], x3[0:500], color='g')
            ax[2].set_ylabel('X3')
            # ax[3].plot(t[0:1000], xs[0][0:1000], color='k')
            # ax[3].set_ylabel(r'$(X1 + X2)^2 + X3^2$ 100Hz')
            ax[3].plot(t[0:500], x[0:500], color='k')
            ax[3].set_ylabel(r'$(X1 + X2)^2 + X3^2$')
            plt.show(block=False)

            xs.append(x)

        x = np.vstack(xs)
        ### project to high-D space 
        highd_proj, log_means = create_highd_space(x, num_neurons)

        # test our high-D projection function 
        spikes, true_rates = project_to_highd(x, highd_proj, log_means)

        fig, ax= plt.subplots(4, 1)
        ax[0].plot(true_rates[0:500, 0])
        ax[0].set_ylabel('Neuron 0 Firing Rate')
        ax[1].plot(spikes[0:500, 0])
        ax[1].set_ylabel('Neuron 0 Spikes')
        ax[2].plot(true_rates[0:500, 57])
        ax[2].set_ylabel('Neuron 57 Firing Rate')
        ax[3].plot(spikes[0:500, 57])
        ax[3].set_ylabel('Neuron 57 Spikes')
        plt.show(block=False)

        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(true_rates[0:150].T)
        ax[1].imshow(spikes[0:150].T)
        plt.show(block=False)

        # xx = spikes[0:99900,:]
        # xx_r = np.reshape(xx, (666, 150, 100)) # trials x time x neurons
        # xx_m = np.mean(xx_r, axis=(0))
        # plt.figure()
        # plt.imshow(xx_m.T)
        # plt.show(block=False)

        # # smoothed spikes 
        # from scipy import signal
        # gauss_width = 45.
        # gauss_bin_std = gauss_width / (1000.0 * dt)
        # # the window extends 3 x std in either direction
        # win_len = int(6 * gauss_bin_std)

        # # MRK : a much faster implementation than pandas rolling
        # window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        # window /=  np.sum(window)
        # spike_vals = [spikes[:,i] for i in range(spikes.shape[1])]

        # def filt(args):
        #     """ MRK: this is used to parallelize spike smoothing
        #     Parallelized function for filtering spiking data
        #     """
        #     x, window = args
        #     y = signal.lfilter(window, 1., x)
        #     # shift the signal (acausal filter)
        #     shift_len = len(window) //2
        #     y = np.concatenate( [y[shift_len:], np.full(shift_len, np.nan)], axis=0 )
        #     return y

        # y_list = [filt((x, window)) for x in spike_vals]
        # smooth_spk = np.vstack(y_list).T
        # xx = smooth_spk[0:99900,:]
        # xx_r = np.reshape(xx, (666, 150, 100)) # trials x time x neurons
        # xx_m = np.mean(xx_r, axis=(0))
        # plt.figure()
        # plt.imshow(xx_m.T)
        # plt.show(block=False)

        return x, x_comps, spikes, true_rates


    def create_dataset(self, x, x_comps, spikes, true_rates): 
        # construct data dictionary from simulated spikes
        data_dict={'spikes':spikes, 'lowD':x, 'true_rates':true_rates, 'lowD_components': np.hstack(x_comps)}

        # stages of trials
        stage_codes = {1:'trialStart',
                      2:'trialEnd',
                      3:'trial_id',
                      4:'good' 
                      }

        # pre-populate the list of dictionaries (number of trials)
        # map everything to None
        trial_info = [dict(zip(stage_codes.values(), [None]*len(stage_codes))) for _ in range(self.num_trials)]

        start_times = np.linspace(0, spikes.shape[0]-50, self.num_trials)
        end_times = np.linspace(50, spikes.shape[0], self.num_trials)

        for t in range(self.num_trials):
            # subtracting one to make sure that first start time is 0 and ensure proper reconstruction of trials
            trial_info[t]['trialStart'] = start_times[t]
            trial_info[t]['trialEnd'] = end_times[t]
            trial_info[t]['trial_id'] = t+1
            trial_info[t]['good'] = 'good' # dummy variable to be able to select all trials later

        self.init_data_from_dict(data_dict=data_dict, sample_interval=self.dt, trial_info=trial_info)

        print('Data loaded.')
            
# main script for creating lfads data from Rossler data
if __name__ == "__main__":
    d = LTCDataset(name='nonlinearOscillators')
    x, x_comps, spikes, true_rates = d.generate_spikes()
    d.create_dataset(x, x_comps, spikes, true_rates)

    d.make_trials(margins = 0.0, start_name='trialStart', end_name='trialEnd')
    
    #select all of the trials
    d.select_trials('good', {'good':'good'})

    lfads_save_path = '/snel/home/brianna/projects/deep_learning_project/'
    lfads_load_path = '/snel/home/brianna/projects/deep_learning_project/'
    
    chop_params = {'trial_length_s':1, 'trial_olap_s':0, 'bin_size_s': 0.02}      # not chopping data
    valid_ratio = 0.1
    d.init_lfads(lfads_save_path, lfads_load_path, chop_params, valid_ratio)

    data_select_props = {
        'data_type': 'spikes',
        'selection': 'good'
        }

    # use segments argument to run only trial information 
    d.create_lfads_data(run_type='segments', data_select_props=data_select_props)
    d.pickle_me(lfads_save_path, True) # saves the Dataset object in case we want to try again with different params

