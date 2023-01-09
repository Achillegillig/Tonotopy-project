import matplotlib.pyplot as plt # import module inside a function to automatically import it when calling f?
import wave
import numpy as np
import pandas as pd
import os 

# audio_filename = 'syn_sin_1000Hz_-6dBFS_0.5s.wav'
# audio_filename = 'protocol0_sequence_mouse_tone.wav_stim_01.wav'

IC_filename = 'C:\\Users\\achil\\Documents\\Universite\\M2_CNN_local\\Research_Project\\code\\ic_out_stim_01_60dB.txt'
CF_file = 'cf.txt'

fs_ic = 20000

CF = pd.read_csv(CF_file, header=None).values

dir_name = '../code'
files = [file for file in os.listdir(dir_name) if file.startswith('ic_out')]

files = [os.path.join(dir_name, file) for file in files]

IC = {}

for filename in files:

    IC_data = pd.read_csv(filename).values.T
    IC_data_avg = []
    #average to have n channels
    n_channels = 10

    avg_factor = int(np.size(IC_data,0)/n_channels)
    for i in range(10):
        IC_data_avg.append((np.mean(IC_data[i*avg_factor:(i+1)*avg_factor], axis=0)))    

    # # from 1000 to 10 linearly spaced channels

    #try with 10 individual channels
    # channels = np.linspace(0,np.size(IC_data,0)-1,10)
    # IC_data_10ch = [IC_data[int(i)] for i in channels]
    # CF_10ch = [CF[int(i)] for i in channels]
    # IC_data_avg = IC_data_10ch

    # print(filename[23:-4])
    IC[filename[23:-4]] = IC_data_avg

IC.pop('-10dB')
n_channels = len(IC.keys())

CF_avg = [CF[int(i*avg_factor)] for i in range(10)]
CF_avg_bis = np.zeros_like(CF_avg)  
for i in range(len(CF_avg)-1):
    CF_avg_bis[i] = CF_avg[i+1]

CF_avg = (CF_avg + CF_avg_bis) /2

SPL_vals = list(IC.keys())

channel_labels = ['{:0>2}'.format(n) for n in range(1,n_channels+1)]
channels_data = {}

for i, channel in zip(range(n_channels), channel_labels):
    temp = []
    for spl in SPL_vals:
        temp.append(np.array(IC[spl])[i,:])

    channels_data[channel] = temp

# """Generate a spike train. See page 5 and 6 here:
# https://www.cns.nyu.edu/~david/handouts/poisson.pdf

# We used the first method described.
# """
# # Seed the generator for consistent results.
np.random.seed(42)

# # We need to start with some sort of r(t). Presumably the asker of this
# # question has access to this.
r = 10 #fixed r: homogeneous Poisson generator
threshold = 0.7
# # Define time interval, delta t. Use one millisecond as suggested
# # in the paper.
dt = 1e-4 # same time interval as IC model

# # Initialize output.
spiketrain = {}

fig = plt.figure(figsize=(45,45/5))
xmin=780
xmax = 3000

subset = np.array([0, 3, 4, 6])
n_channels = len([channel_labels[j] for j in subset])

for i, channel in zip(range(n_channels), [channel_labels[j] for j in subset]):
# for i, channel in zip(range(n_channels), channel_labels):    

    # ##scale data to be [0 1]
    channels_data[channel] = np.array(channels_data[channel])
    channels_data_scaled = [(channels_data[channel][i,:] - np.min(channels_data[channel][i,:])) / (np.max(channels_data[channel][i,:]) - np.min(channels_data[channel][i,:])) for i in range(len(IC.keys()))]
    #select only the first stim of the burst for better temporal resolution
    channels_data_scaled = np.array(channels_data_scaled)[:,xmin:xmax]

    #stim onset : ~0.0780s // offset 0.12s -> x[int(0.0780*fs_ic):int(0.12*fs_ic)]

    #spike train generation
    # mask = np.array(channels_data[channel]) >= r * dt
    mask = np.array(channels_data_scaled) >= threshold
    spikes = np.zeros_like(channels_data_scaled)
    spikes[mask] = 1

    spiketiming = np.nonzero(spikes)
    active_channels = np.unique(spiketiming[0])
    spiketiming = np.split(spiketiming[1], np.unique(spiketiming[0], return_index=True)[1][1:])
    
    spike_times = []
    c = 0
    for j in range(len(SPL_vals)):
        if j in active_channels:
            spike_times.append(spiketiming[c])
            c+=1
        else:
            spike_times.append([])
            
    spiketiming = spike_times

    ax = plt.subplot(1, int(n_channels), i+1)
    plt.eventplot(spiketiming, colors='black', linelengths=0.1, linewidths=1, lineoffsets=1)
    # plt.xlim(xmin, xmax)
    plt.rc('axes', titlesize=40)
    plt.rc('axes', labelsize=30)
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    if i == 0:
        plt.xlabel('Time after stim. onset (s)')
        plt.ylabel('Sound pressure level')

    xticks = np.linspace(0,xmax-1,5, dtype=int)
    xticklabels = ["{:.2f}".format(float((i/20000))) for i in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = np.linspace(0,len(SPL_vals)-1,len(IC.keys()), dtype=int)
    yticklabels = SPL_vals

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.title.set_text('# ' + channel + ', mean CF: ' + "{:.2f}".format(float(CF_avg[i]/1000)) + ' kHz')


fig.tight_layout()
fig.subplots_adjust(wspace=0.3)