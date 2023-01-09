import matplotlib.pyplot as plt # import module inside a function to automatically import it when calling f?
import wave
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

audio_filename = 'protocol0_sequence_mouse_tone_stim_01.wav'

IC_filename = 'C:\\Users\\achil\\Documents\\Universite\\M2_CNN_local\\Research_Project\\code\\ic_out_stim_01_60dB.txt'
CF_file = 'cf.txt'

fs_ic = 20000 #(parameter of  AMToolbox output)

wavdata = None
with wave.open(audio_filename, mode='rb') as wf:
    # Read the whole file into a buffer (for samll files).
    buffer = wf.readframes(wf.getnframes())
    # Convert the buffer to a numpy array by checking the size of the sample in bytes.
    # The output will be a 1D array with interleaved channels.
    interleaved = np.frombuffer( buffer, dtype = 'int%d'%(wf.getsampwidth()*8) )
    # interleaved = interleaved/interleaved.max()
    # Reshape it into a 2D array separating the channels in columns.
    wavdata = np.reshape(interleaved, (-1, wf.getnchannels()))
    wavdata = wavdata.copy()
    

fig1 = plt.figure(figsize=(40,10))

wavdata[wavdata==0] = np.random.randint(1,20, size=wavdata[wavdata==0].shape)

ax1 = plt.subplot(131)
ax1.title.set_text('Stimulus spectrogram')

Pxx, freqs, dt, im = plt.specgram(wavdata.flatten(), NFFT=128, Fs=wf.getframerate(), noverlap=2, scale='dB', cmap='viridis') # (43920,)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.subplots_adjust(wspace=1.2)

cbarticks = np.linspace(0,np.size(IC_data_avg,1),5, dtype=int)
cbar = plt.colorbar(shrink=0.6, ticks=ticker.LinearLocator(5))
cbar.set_label('Intensity (dB)', rotation=270)

CF = pd.read_csv(CF_file, header=None)
CF = np.array(CF.values)

IC_data = pd.read_csv(IC_filename, header=None)
IC_data = IC_data.values
IC_data = IC_data.T

###Different methods to go from 1000 to 10 channels can be employed
####have been tested

# from 1000 to 10 linearly spaced channels
# channels = np.linspace(0,np.size(IC_data,0)-1,10)
# IC_data_10ch = [IC_data[int(i)] for i in channels]
# CF_10ch = [CF[int(i)] for i in channels]


#average to have n channels
n_channels = 10
IC_data_avg = []
avg_factor = int(np.size(IC_data,0)/n_channels)
for i in range(10):
    IC_data_avg.append((np.mean(IC_data[i*avg_factor:(i+1)*avg_factor], axis=0)))


CF_avg = [CF[int(i*avg_factor)] for i in range(10)]

CF_avg_bis = np.zeros_like(CF_avg)
     
for i in range(len(CF_avg)-1):
    CF_avg_bis[i] = CF_avg[i+1]

CF_avg = (CF_avg + CF_avg_bis) / 2

ax4 = plt.subplot(132)

plt.imshow(IC_data_avg, aspect='1000', cmap='viridis')

cbar = plt.colorbar(shrink=0.6, ticks=ticker.LinearLocator(5))
# cbar.set_label('Rate (spikes/s)', rotation=270)

xticks = np.linspace(0,np.size(IC_data_avg,1),5, dtype=int)
xticklabels = ["{:.2f}".format(float((i/20000))) for i in xticks]

ax4.set_xticks(xticks)
ax4.set_xticklabels(xticklabels)

# ax2.set_ylim(ax1.get_ylim()) #align plot2 to plot1
yticks = np.linspace(0,np.size(IC_data_avg,0)-1,10, dtype=int)
yticklabels = ["{:.2f}".format(float(CF_avg[i]/1000)) for i in yticks]

ax4.set_yticks(yticks)
ax4.set_yticklabels(yticklabels)
# ax4.invert_yaxis()
ax4.title.set_text('IC rate, 10 channels')

plt.ylabel('mean CF (kHz)')
plt.xlabel('Time (s)')

# Generate spike trains from IC out


"""Generate a spike train. See page 5 and 6 here:
https://www.cns.nyu.edu/~david/handouts/poisson.pdf

We used the first method described.
"""
# Seed the generator for consistent results.
np.random.seed(42)

threshold = 0.6

spikes = np.zeros_like(IC_data_avg)


IC_data_scaled = []
# IC_data_scaled[IC_data_scaled < 0] = 0

##scale data to be [0 1]
IC_data_avg = np.array(IC_data_avg)
IC_data_scaled = [(IC_data_avg[i,:] - np.min(IC_data_avg[i,:])) / (np.max(IC_data_avg[i,:]) - np.min(IC_data_avg[i,:])) for i in range(len(IC_data_avg))]


#Generate spikes
# mask = np.array(IC_data_scaled) >= r * dt
mask = np.array(IC_data_scaled) >= threshold
spikes[mask] = 1

spiketiming = np.nonzero(spikes)
active_channels = np.unique(spiketiming[0])
spiketiming = np.split(spiketiming[1], np.unique(spiketiming[0], return_index=True)[1][1:])

spike_times = []
c = 0
for j in range(n_channels):
    if j in active_channels:
        spike_times.append(spiketiming[c])
        c+=1
    else:
        spike_times.append([])
        c+=1

spiketiming = spike_times

ax3 = plt.subplot(133)

plt.eventplot(spiketiming, colors='black', linelengths=0.1, linewidths=0.5, lineoffsets=1)
plt.xlim(0,len(IC_data_avg[1,:]))

ax3.set_yticks(yticks)
ax3.set_yticklabels(yticklabels)
ax3.invert_yaxis()
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticklabels)

ax3.title.set_text('IC spike trains, 10 channels')

plt.ylabel('mean CF (kHz)')
plt.xlabel('Time (s)')