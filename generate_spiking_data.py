import numpy as np 
import matplotlib.pyplot as plt 
from rds.structures import Dataset

class LTCDataset(Dataset): 
    def generate_spikes(self):
        ### define the parameters of the simulation
        trial_length = 10 # length of each trial, s 
        self.dt = 0.01 # timestep, s
        dt = self.dt
        self.num_trials = 2000 # number of trials to simulate 
        num_trials = self.num_trials
        num_ics = 4 # number of initial conditions to sample 
        ics = [(1,0), (0, -1), (-1, 0), (0, 1)] # the set of initial conditions (x1, x2)

        period = 1 # oscillator period 
        amplitudes = range(1,5) # oscillator amplitude 

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

        ### generate oscillators with three different sets of parameters 
        x1 = oscillator(length, dt, period=1, amplitude=1, ic=1)
        x2 = oscillator(length, dt, period=1.5, amplitude=1, ic=0)
        x3 = oscillator(length, dt, period=0.5, amplitude=1, ic=0.5)

        ### visualize components of compound system 
        t = np.linspace(0, length, int(length / dt))
        fig, ax = plt.subplots(3,1, sharex=True, sharey=True)
        ax[0].plot(t[0:1000], x1[0:1000])
        ax[1].plot(t[0:1000], x2[0:1000])
        ax[2].plot(t[0:1000], x3[0:1000])
        plt.show(block=False)

        ### perform a nonlinear combination of the three above components and plot 
        x = (x1 + x2)**2 + x3**2

        fig, ax = plt.subplots(4,1)
        ax[0].plot(t[0:1000], x1[0:1000], color='b')
        ax[0].set_ylabel('X1')
        ax[1].plot(t[0:1000], x2[0:1000], color='r')
        ax[1].set_ylabel('X2')
        ax[2].plot(t[0:1000], x3[0:1000], color='g')
        ax[2].set_ylabel('X3')
        ax[3].plot(t[0:1000], x[0:1000], color='k')
        ax[3].set_ylabel(r'$(X1 + X2)^2 + X3^2$')
        plt.show(block=False)

        ### project to high-D space 
        def create_highd_space(lowd, num_neurons):

            # don't care what the high-D space is, so use random
            # balance it by subtracting 0.5
            highd_proj = np.random.rand(lowd.shape[1], num_neurons)-0.5

            # each neuron can also have a different mean "rate"
            # assign these randomly
            log_means = (np.random.rand(num_neurons) - 0.5) * 0.001

            return highd_proj, log_means

        highd_proj, log_means = create_highd_space(x, num_neurons)

        def project_to_highd(lowd, highd_proj, log_means):
            # project the low-D data out to high-D space
            z_true = np.dot(lowd, highd_proj) + log_means

            # exponential nonlinearity to convert these to firing rates 
            z_true_rates = np.exp(z_true)

            spikes = np.random.poisson(z_true_rates*dt)

            return spikes, z_true_rates

        # test our high-D projection function 
        spikes, true_rates = project_to_highd(x, highd_proj, log_means)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.imshow(spikes[0:1000].T, extent=[0, 10, 0, 100], cmap='binary', vmax=0.1)
        plt.xlabel('Time (s)')
        plt.ylabel('"Neurons"')
        plt.show(block=False)

        return x, spikes, true_rates


    def create_dataset(self, x, spikes, true_rates): 
        # construct data dictionary from simulated spikes
        data_dict={'spikes':spikes, 'lowD':x, 'true_rates':true_rates}

        # stages of trials
        stage_codes = {1:'trialStart',
                      2:'trialEnd',
                      3:'trial_id',
                      4:'good' 
                      }

        # pre-populate the list of dictionaries (number of trials)
        # map everything to None
        trial_info = [dict(zip(stage_codes.values(), [None]*len(stage_codes))) for _ in range(self.num_trials)]

        start_times = np.linspace(0, spikes.shape[0]-1000, self.num_trials)
        end_times = np.linspace(1000, spikes.shape[0], self.num_trials)

        for t in range(0, self.num_trials):
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
    x, spikes, true_rates = d.generate_spikes()
    d.create_dataset(x, spikes, true_rates)

    # # rebin into 10ms bins 10/1/19
    # d.resample(target_bin=0.01)

    d.make_trials(margins = 0.0, start_name='trialStart', end_name='trialEnd')
    
    #select all of the trials
    d.select_trials('good', {'good':'good'})

    lfads_save_path = '/snel/home/brianna/projects/deep_learning_project/'
    lfads_load_path = '/snel/home/brianna/projects/deep_learning_project/'
    
    chop_params = {'trial_length_s':1, 'trial_olap_s':0, 'bin_size_s': 0.01}      # not chopping data
    valid_ratio = 0.2
    d.init_lfads(lfads_save_path, lfads_load_path, chop_params, valid_ratio)

    data_select_props = {
        'data_type': 'spikes',
        'selection': 'good'
        }

    # use segments argument to run only trial information 
    d.create_lfads_data(run_type='segments', data_select_props=data_select_props)
    d.pickle_me(lfads_save_path, True) # saves the Dataset object in case we want to try again with different params

