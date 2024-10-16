import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.patches import Circle

import dimensionality as dim
import utils as utils
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_random_stim import RandomStimElectrodes
from network_cortcol import Network

# matplotlib.use("Agg")

# Cortical Microcircuit Simulation-specific settings

# ====================PROBE=====================================================
# Layout: 1x32 spans 1800um
volume_v_min = 200  # um
ch_vcoords = np.arange(32) * 60
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
# stim configurations
sigma_e_um = 2.76e-7
conductivity_constant = 60  # tune this

PRESIM_TIME_MS = sim_dict["t_presim"]
SIM_TIME_MS = sim_dict["t_sim"]
WINDOW_MS = 500
OVERLAP_MS = 400
RASTER_PLOT_TIME_MS = 1000
N_GROUPS_LIST = [1, 2, 4, 8, 16, 32]  # number of stim electrode groups
RASTER_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + RASTER_PLOT_TIME_MS]
FIRING_RATE_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + SIM_TIME_MS]

""" 
Calculates the decayed amplitude of the induced intracellular stimulation current 
based on the distance from the stimulation electrode. Takes into account a small offset
and conductivity constant. 
"""


def amp_decay_func(amp_uA, dist_um):
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )


stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}

nest.ResetKernel()
sim_dict["data_path"] = os.path.join(os.getcwd(), "data20241011/data_baseline20241011")
if os.path.exists(sim_dict["data_path"]):
    shutil.rmtree(sim_dict["data_path"])

network = Network(sim_dict, net_dict, stim_dict)
network.create()
network.connect()
print("Total # neurons simulated:", network.n_neurons)

# Baseline simulation
network.simulate_baseline(PRESIM_TIME_MS)  # startup transient

network.simulate_baseline(SIM_TIME_MS)  # data collected after pre-simulation

network.evaluate(
    raster_plot_interval=RASTER_INTERVAL,
    firing_rates_interval=FIRING_RATE_INTERVAL,
    title="Baseline activity",
)

# %% Process baseline recording
baseline_spike_trains = network.get_spike_train_list()
baseline_spike_rates = dim.compute_spike_rates(
    baseline_spike_trains,
    SIM_TIME_MS,
    WINDOW_MS,
    OVERLAP_MS,
    presim_time_ms=PRESIM_TIME_MS,
)
baseline_pca, pca_model = dim.fit_and_transform_baseline_pca(baseline_spike_rates)

# %%
# Random stimulation
# for n_groups in [32, 8, 4, 2]:
#     nest.ResetKernel()
#     electrodes = RandomStimElectrodes(
#         ch_coordinates, stim_pulse_params, amp_decay_func)
#     sim_dict["data_path"] = os.path.join(
#         os.getcwd(), f"data20241011/data_randstim_{n_groups}groups/"
#     )
#     network = Network(sim_dict, net_dict, stim_dict)
#     network.create()
#     network.connect()
#     stim_channels = np.arange(32)
#     amplitude_range = [
#         1,
#         5,
#     ]  # larger amplitude range and larger amplitudes lead to higher dimensionality
#     stim_rate_hz = 20  # higher stim rate leads to lower dimensionality
#     # n_groups = 1 # number of electrode groups (grouped by spatial location)
#     electrodes.generate_random_stimulation(
#         stim_channels,
#         amplitude_range,
#         SIM_TIME_MS,
#         stim_rate_hz=stim_rate_hz,
#         n_groups=n_groups,
#     )
#     currents_at_neurons = electrodes.get_current_at_locs(
#         network.neuron_locations)

#     network.simulate_current_input(currents_at_neurons, PRESIM_TIME_MS)
#     network.simulate_current_input(currents_at_neurons, SIM_TIME_MS)

#     network.evaluate(RASTER_INTERVAL, FIRING_RATE_INTERVAL)
#     plt.figure()
#     ax = plt.subplot(111)
#     for i in range(len(ch_coordinates)):
#         ax.add_patch(
#             Circle(ch_coordinates[i, :], 12.5, fill=False, edgecolor="k"))
#     for p in stim_channels:
#         ax.scatter(*ch_coordinates[p, :], c="r", marker="x", s=12.5)
#     for n in range(network.n_neurons):
#         ax.scatter(
#             *network.neuron_locations[n, :],
#             c="green",
#             marker="s",
#             s=2.5,
#             alpha=0.1,
#         )
#     ax.set_aspect("equal")
#     plt.savefig(os.path.join(
#         sim_dict["data_path"], f"probe_{n_groups}groups.png"))
#     plt.close()

#     plt.figure()
#     ax = plt.subplot(111)
#     electrodes.plot_stim_raster(ax=ax, time_range_ms=[0, tsim_ms])
#     plt.savefig(
#         os.path.join(sim_dict["data_path"],
#                      f"stim_raster_{n_groups}groups.png")
#     )
#     plt.close()

#     plt.figure()
#     ax = plt.subplot(111)
#     stamps = network.get_spktrains()
#     for k, v in stamps.items():
#         ax.plot(v, np.ones_like(v) * k, "k.", markersize=0.1)
#     plt.savefig(
#         os.path.join(sim_dict["data_path"],
#                      f"tmp_stamps_corcol_{n_groups}groups.png")
#     )
#     plt.close()
