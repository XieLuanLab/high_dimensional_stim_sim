import os

import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('Agg')

from network_cortcol import Network
import utils as utils
import dimensionality as dim
from electrodes_random_stim import RandomStimElectrodes
# Cortical Microcircuit Simulation-specific settings
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from corcol_params.network_params import net_dict

# ====================PROBE=====================================================
# Layout: 1x32 spans 1800um
volume_v_min = 200 # um
ch_vcoords = np.arange(32) * 60
ch_hcoords = np.zeros(32) # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
# stim configurations
sigma_e_um = 2.76e-7
conductivity_constant = 60 # tune this
amp_decay_func = lambda amp_uA, dist_um: amp_uA * 1e-6 * conductivity_constant / (4 * np.pi * sigma_e_um * (dist_um + 20))
stim_pulse_params = {
    "pulse_width_ms": 0.2,
    "ipi_ms": 0.2
}

# Network
nest.ResetKernel()
sim_dict["data_path"] = os.path.join(os.getcwd(), 'data20241011/data_baseline20241011')
network = Network(sim_dict, net_dict, stim_dict)
network.create()
network.connect()
print("Total #neurons actually simulated:", network.n_neurons)
# exit(0)
# baseline
tsim_ms = sim_dict["t_presim"]+sim_dict["t_sim"]
# tbin_ms = 20
network.simulate_baseline(tsim_ms)
network.evaluate([0, tsim_ms], [0, tsim_ms])

# Random stimulation
for n_groups in [32, 8, 4, 2]:
    nest.ResetKernel()
    electrodes = RandomStimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)
    sim_dict["data_path"] = os.path.join(os.getcwd(), f'data20241011/data_randstim_{n_groups}groups/')
    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    stim_channels = np.arange(32)
    amplitude_range = [1,5] # larger amplitude range and larger amplitudes lead to higher dimensionality
    stim_rate_hz = 20 # higher stim rate leads to lower dimensionality
    # n_groups = 1 # number of electrode groups (grouped by spatial location)
    electrodes.generate_random_stimulation(
        stim_channels, amplitude_range, tsim_ms,
        stim_rate_hz=stim_rate_hz, n_groups=n_groups)
    currents_at_neurons = electrodes.get_current_at_locs(network.neuron_locations)
    network.simulate_current_input(currents_at_neurons, tsim_ms)
    network.evaluate([0, tsim_ms], [0, tsim_ms])
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(ch_coordinates)):
        ax.add_patch(Circle(ch_coordinates[i, :], 12.5, fill=False, edgecolor="k"))
    for p in stim_channels:
        ax.scatter(*ch_coordinates[p, :], c="r", marker="x", s=12.5)
    for n in range(network.n_neurons):
        ax.scatter(*network.neuron_locations[n, :], c="green", marker="s", s=2.5, alpha=0.1)
    ax.set_aspect("equal")
    plt.savefig(os.path.join(sim_dict["data_path"], f"probe_{n_groups}groups.png"))
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    electrodes.plot_stim_raster(ax=ax, time_range_ms = [0, tsim_ms])
    plt.savefig(os.path.join(sim_dict["data_path"], f"stim_raster_{n_groups}groups.png"))
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    stamps = network.get_spktrains()
    for k, v in stamps.items():
        ax.plot(v, np.ones_like(v)*k, 'k.', markersize=0.1)
    plt.savefig(os.path.join(sim_dict["data_path"], f"tmp_stamps_corcol_{n_groups}groups.png"))
    plt.close()
