import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from electrodes import Electrodes
from network import Network

# initialize numpy random generator
rng = np.random.default_rng(41)
# initalize nest
nest.ResetKernel()
nest.resolution = 0.01 # ms
nest.rng_seed = 41

# 16x2 zig zag pattern
ch_vcoords = np.arange(32)*20 # um
ch_hcoords = np.zeros(32) # um
ch_hcoords[1::2] += 25
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
probe_hmin = ch_coordinates[:, 0].min()
probe_hmax = ch_coordinates[:, 0].max()
probe_vmin = ch_coordinates[:, 1].min()
probe_vmax = ch_coordinates[:, 1].max()

# neuron locations (all neurons are within probe)
n_neurons = 100
neuron_cooridnates = rng.random((n_neurons, 2), dtype=float)
neuron_cooridnates[:, 0] = neuron_cooridnates[:, 0] * (probe_hmax - probe_hmin) + probe_hmin
neuron_cooridnates[:, 1] = neuron_cooridnates[:, 1] * (probe_vmax - probe_vmin) + probe_vmin

sigma_um = 100
amp_decay_func = lambda amp, dist: amp * np.exp(-dist**2/(2*sigma_um**2))
electrodes = Electrodes(ch_coordinates=ch_coordinates, current_disperse_func=amp_decay_func)
connectivity_matrix = (rng.random((n_neurons, n_neurons)) - 0.5) * 200
connectivity_matrix[np.abs(connectivity_matrix) < np.quantile(np.abs(connectivity_matrix), .20)] = 0
nn = Network(n_neurons, neuron_cooridnates, connectivity_matrix)

stim_ch_params = []
stim_ch_params.append({
    "ch_idx": 0,
    "ampl_ua": 5,
    "freq_hz": 20,
    "npulses": 10,
    "ipi_ms": 0.1,
    "pulsewidth_ms": 0.2,
    "onset_time_ms": 20
})
electrodes.set_biphasic_pulsetrain(stim_ch_params)
currents_at_neurons = electrodes.get_current_at_locs(nn.neuron_locations)

sim_time = 500
nn.simulate_current_input(currents_at_neurons, sim_time)

spiketrains = nn.get_spiketrains()
neuron_inds_sorted = np.argsort(nn.neuron_locations[:, 1])
plt.figure()
for i in range(n_neurons):
    idx = neuron_inds_sorted[i]
    spiketrain = spiketrains[idx]
    plt.eventplot(spiketrain, lineoffsets=i, colors="k")#, linestyles="|")
plt.xlim(0, sim_time)
plt.ylim(0, n_neurons+1)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index (depth-reordered)")
plt.savefig("spiketrains.png")
plt.close()

volt_traces = nn.get_voltages()
plt.figure(figsize=(40,20))
for i in range(n_neurons):
    ax = plt.subplot(10, 10, i+1)
    voltage_times, voltage_trace = volt_traces[i]
    plt.plot(voltage_times, voltage_trace, label=f"Neuron {i}", color="k")
plt.savefig("voltages.png")
plt.close()

plt.figure()
ax = plt.subplot(111)
for i in range(len(ch_coordinates)):
    ax.add_patch(Circle(ch_coordinates[i, :], 12.5, fill=False, edgecolor="k"))
for p in stim_ch_params:
    ax.scatter(*ch_coordinates[p['ch_idx'], :], c="b", marker="x", s=12.5)
for n in range(n_neurons):
    ax.scatter(*neuron_cooridnates[n, :], c="green", marker="s", s=12.5)
ax.set_aspect("equal")
# plt.show()
plt.savefig("probe.png")
plt.close()
