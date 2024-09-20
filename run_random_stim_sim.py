from electrodes_random_stim import RandomStimElectrodes
import nest
import numpy as np
from network import Network
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import utils as utils
import dimensionality as dim
#%%
n_neurons = 200
rng = np.random.default_rng(42)
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

neuron_coordinates = rng.random((n_neurons, 2), dtype=float)
neuron_coordinates[:, 0] = neuron_coordinates[:, 0] * (volume_h_max - volume_h_min) + volume_h_min
neuron_coordinates[:, 1] = neuron_coordinates[:, 1] * volume_v_min

connectivity_matrix = (rng.random((n_neurons, n_neurons)) - 0.5) * 200
connectivity_matrix[np.abs(connectivity_matrix) < np.quantile(np.abs(connectivity_matrix), .20)] = 0

sigma_e_um = 2.76e-7
conductivity_constant = 60 # tune this
amp_decay_func = lambda amp_uA, dist_um: amp_uA * 1e-6 * conductivity_constant / (4 * np.pi * sigma_e_um * (dist_um + 20)) 

stim_pulse_params = {
    "pulse_width_ms": 0.2,
    "ipi_ms": 0.2
}

#%% Simulate baseline
nest.ResetKernel()
sim_time_ms = 30000 
nn = Network(n_neurons, neuron_coordinates, connectivity_matrix)

nn.simulate_baseline(sim_time_ms)
baseline_spike_trains = nn.get_spiketrains()

# Visualize baseline results
baseline_spike_rates = dim.get_dimensionality(baseline_spike_trains)
utils.plot_raster(nn, time_range_ms=[0, sim_time_ms/20])

#%% Simulate stimulation
nest.ResetKernel()
electrodes = RandomStimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)

sim_time_ms = 30000 
stim_channels = np.arange(32)
amplitude_range = [1,5] # larger amplitude range and larger amplitudes lead to higher dimensionality
stim_rate_hz = 20 # higher stim rate leads to lower dimensionality
n_groups = 1 # number of electrode groups (grouped by spatial location)

electrodes.generate_random_stimulation(stim_channels, amplitude_range, 
                                       sim_time_ms, stim_rate_hz, n_groups)


nn = Network(n_neurons, neuron_coordinates, connectivity_matrix)

currents_at_neurons = electrodes.get_current_at_locs(nn.neuron_locations)

nn.simulate_current_input(currents_at_neurons, sim_time_ms)
spike_trains = nn.get_spiketrains()
# Visualize stim results
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
utils.plot_raster(nn, time_range_ms=[0, sim_time_ms/20], ax=axes[0])
electrodes.plot_stim_raster(ax=axes[1], time_range_ms=[0, sim_time_ms/20])
plt.tight_layout()
utils.plot_firing_rates(nn, time_range_ms=[0, sim_time_ms/20], plot_top_n=10)
spike_rates = dim.get_dimensionality(spike_trains, sim_time_ms)
#%% Plot dimensionality vs stim channels

# Manually fill in dim values
dims = [36, 43, 49, 52, 57, 57]  # Dimensionality values
stim_channel_cnts = [1, 2, 4, 8, 16, 32]  # Number of stimulation channels

plt.figure(figsize=(8, 5))
plt.plot(stim_channel_cnts, dims, marker='o', label="Evoked Dimensionality")
plt.axhline(60, color='r', linestyle='--', label="Baseline Dimensionality")  # Add baseline reference line

# Set x-axis tick labels
plt.xticks(stim_channel_cnts, labels=stim_channel_cnts)

# Add labels and title
plt.xlabel('Number of Stimulated Channels')
plt.ylabel('Dimensionality')
plt.title('Stimulus-Evoked vs Baseline Dimensionality')
plt.legend()

# Show plot
plt.show()

#%% Plot channels and neurons
plt.figure()
ax = plt.subplot(111)
for i in range(len(ch_coordinates)):
    ax.add_patch(Circle(ch_coordinates[i, :], 12.5, fill=False, edgecolor="k"))
for p in stim_channels:
    ax.scatter(*ch_coordinates[p, :], c="r", marker="x", s=12.5)
for n in range(n_neurons):
    ax.scatter(*neuron_coordinates[n, :], c="green", marker="s", s=12.5)
ax.set_aspect("equal")
plt.show()
