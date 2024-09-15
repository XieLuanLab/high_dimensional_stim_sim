import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from electrodes import Electrodes
from network import Network
import utils as utils

# initialize numpy random generator
rng = np.random.default_rng(42)
# initalize nest
nest.ResetKernel()
nest.resolution = 0.01 # ms
nest.rng_seed = 45

# 16x2 zig zag pattern
volume_v_min = 1000 # um
ch_vcoords = np.arange(32) * 20  
ch_vcoords = ch_vcoords + volume_v_min - np.max(ch_vcoords)
ch_vcoords = ch_vcoords + volume_v_min - np.max(ch_vcoords) 
ch_hcoords = np.zeros(32) # um
ch_hcoords[1::2] += 25

ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
probe_hmin = ch_coordinates[:, 0].min()
probe_hmax = ch_coordinates[:, 0].max()
probe_vmin = ch_coordinates[:, 1].min()
probe_vmax = ch_coordinates[:, 1].max()

probe_h_center = (probe_hmin + probe_hmax) / 2

volume_h_offset = 400 # um
volume_h_max = probe_h_center + volume_h_offset
volume_h_min = probe_h_center - volume_h_offset

# neuron locations (all neurons are within probe)
n_neurons = 200
neuron_coordinates = rng.random((n_neurons, 2), dtype=float)
neuron_coordinates[:, 0] = neuron_coordinates[:, 0] * (volume_h_max - volume_h_min) + volume_h_min
neuron_coordinates[:, 1] = neuron_coordinates[:, 1] * volume_v_min

sigma_e_um = 2.76e-7
conductivity_constant = 100 # tune this to match experimental results
amp_decay_func = lambda amp_uA, dist_um: amp_uA * 1e-6 * conductivity_constant / (4 * np.pi * sigma_e_um * (dist_um + 20)) 

sigma_um = 100
amp_decay_func_original = lambda amp, dist: amp * np.exp(-dist**2/(2*sigma_um**2))

distance_vec = np.arange(1, 500)
plt.plot(distance_vec, amp_decay_func_original(5, distance_vec), label="Gaussian")
plt.plot(distance_vec, amp_decay_func(5, distance_vec), label="1/r decay")
plt.legend()
plt.ylim([0, 10])
plt.ylabel("Induced intracellular current (pA)")
plt.xlabel("Distance from electrode (um)")
#%%
nest.ResetKernel()
electrodes = Electrodes(ch_coordinates=ch_coordinates, current_disperse_func=amp_decay_func)
connectivity_matrix = (rng.random((n_neurons, n_neurons)) - 0.5) * 200
connectivity_matrix[np.abs(connectivity_matrix) < np.quantile(np.abs(connectivity_matrix), .20)] = 0
nn = Network(n_neurons, neuron_coordinates, connectivity_matrix)

stim_ch_params = []
ampl_ua = 10
stim_ch_params.append({
    "ch_idx": 0,
    "ampl_ua": ampl_ua,
    "freq_hz": 100,
    "npulses": 50,
    "ipi_ms": 0.1,
    "pulsewidth_ms": 0.2,
    "onset_time_ms": 2000
})

stim_ch_params.append({
    "ch_idx": 15,
    "ampl_ua": ampl_ua,
    "freq_hz": 100,
    "npulses": 50,
    "ipi_ms": 0.1,
    "pulsewidth_ms": 0.2,
    "onset_time_ms": 1500
})

stim_ch_params.append({
    "ch_idx": 31,
    "ampl_ua": ampl_ua,
    "freq_hz": 100,
    "npulses": 50,
    "ipi_ms": 0.1,
    "pulsewidth_ms": 0.2,
    "onset_time_ms": 1000
})

electrodes.set_biphasic_pulsetrain(stim_ch_params)

currents_at_neurons = electrodes.get_current_at_locs(nn.neuron_locations)

sim_time = 5000
nn.simulate_current_input(currents_at_neurons, sim_time)
spiketrains = nn.get_spiketrains()
#%%

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
utils.plot_raster(nn, ax=axes[0])

stim_channel_indices = list(electrodes.stim_onset_times_by_ch.keys())
num_stim_channels = len(stim_channel_indices)
for i, ch_idx in enumerate(stim_channel_indices):
    stim_train = electrodes.stim_onset_times_by_ch[ch_idx]
    axes[1].eventplot(stim_train, lineoffsets=i + 1, linelengths=0.5, colors="k")  # Plot stim trains

axes[1].set_xlim(0, sim_time)
axes[1].set_ylim(0, num_stim_channels + 1)
axes[1].set_yticks(range(1, num_stim_channels + 1))
axes[1].set_yticklabels([str(ch_idx) for ch_idx in stim_channel_indices])
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Stim channel index")

plt.figure()
utils.plot_firing_rates(nn, plot_top_n=10)

#%%
plt.figure()
ax = plt.subplot(111)
for i in range(len(ch_coordinates)):
    ax.add_patch(Circle(ch_coordinates[i, :], 12.5, fill=False, edgecolor="k"))
for p in stim_ch_params:
    ax.scatter(*ch_coordinates[p['ch_idx'], :], c="b", marker="x", s=12.5)
for n in range(n_neurons):
    ax.scatter(*neuron_coordinates[n, :], c="green", marker="s", s=12.5)
ax.set_aspect("equal")

