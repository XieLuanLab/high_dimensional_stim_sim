import os
import pickle
import shutil

import matplotlib.pyplot as plt
import nest
import numpy as np
from sklearn.decomposition import PCA

import dimensionality as dim
import helpers
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
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)  # um
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)
# stim configurations
sigma_e_um = 2.76e-7
conductivity_constant = 10
STIM_CHANNELS = np.arange(32)
STIM_AMPLITUDES = [2]  # uA
STIM_POISSON_RATE_HZ = 8

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


# %%
stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}

nest.ResetKernel()

base_path = os.path.join(os.getcwd(), "data_8Hz_k10_scale01")

sim_dict["data_path"] = os.path.join(base_path, "data_baseline")

pkl_path = os.path.join(sim_dict["data_path"], "baseline_spike_rates.pkl")

if not os.path.exists(pkl_path):
    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    print("Total # neurons simulated:", network.n_neurons)

    # Baseline simulation
    network.simulate_baseline(PRESIM_TIME_MS)  # startup transient
    # data collected after pre-simulation
    network.simulate_baseline(SIM_TIME_MS)

    baseline_spike_trains = network.get_spike_train_list()
    baseline_spike_rates = dim.compute_spike_rates(
        baseline_spike_trains,
        SIM_TIME_MS,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )

    network.evaluate(
        raster_plot_interval=RASTER_INTERVAL,
        firing_rates_interval=FIRING_RATE_INTERVAL,
        title="Baseline activity",
    )
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_dict["data_path"], "baseline.png"))
    plt.close()

    with open(pkl_path, "wb") as f:
        pickle.dump(baseline_spike_rates, f)

with open(pkl_path, "rb") as f:
    baseline_spike_rates = pickle.load(f)
# %% Simulate stimulation

# Check if spike rates saved previously and if so, skip processing

stim_spike_rates_list = []

for n_groups in N_GROUPS_LIST:
    sim_dict["data_path"] = os.path.join(base_path, f"data_randstim_{n_groups}groups/")
    pkl_path = os.path.join(
        sim_dict["data_path"], f"{n_groups}groups_stim_spike_rates.pkl"
    )

    if not os.path.exists(pkl_path):
        print(f"\n\n***** n_groups: {n_groups} *****\n\n")
        nest.ResetKernel()

        electrodes = RandomStimElectrodes(
            ch_coordinates, stim_pulse_params, amp_decay_func
        )

        network = Network(sim_dict, net_dict, stim_dict)
        network.create()
        network.connect()

        network.simulate_baseline(PRESIM_TIME_MS)  # startup transient

        electrodes.generate_random_stimulation(
            STIM_CHANNELS,
            STIM_AMPLITUDES,
            SIM_TIME_MS,
            stim_rate_hz=STIM_POISSON_RATE_HZ,
            n_groups=n_groups,
        )

        # compute impulse response matrix where impulse is stim at 1 uA
        electrodes.compute_impulse_response_matrix(network.neuron_locations)
        electrodes.compute_stim_current_matrix()
        electrodes.calculate_induced_current_matrix()

        current_generators = electrodes.get_current_generators(
            presim_time_ms=PRESIM_TIME_MS
        )

        network.simulate_current_input(
            current_generators, time_ms=SIM_TIME_MS
        )  # stimulate and record

        # Get evoked spike rates and append to list
        stim_evoked_spike_trains = network.get_spike_train_list()
        stim_evoked_spike_rates = dim.compute_spike_rates(
            stim_evoked_spike_trains,
            SIM_TIME_MS,
            WINDOW_MS,
            OVERLAP_MS,
            presim_time_ms=PRESIM_TIME_MS,
        )

        # Spike raster and stim raster
        fig, axes = plt.subplots(2, 1, figsize=(3, 4))
        plot_title = f"{n_groups} electrode group{'s' if n_groups > 1 else ''}"
        network.evaluate(
            RASTER_INTERVAL, FIRING_RATE_INTERVAL, title=plot_title, raster_ax=axes[0]
        )
        electrodes.plot_stim_raster(ax=axes[1], time_range_ms=RASTER_INTERVAL)
        plt.tight_layout()
        plt.savefig(
            os.path.join(sim_dict["data_path"], f"stim_raster_{n_groups}groups.png")
        )
        plt.close()

        with open(pkl_path, "wb") as f:
            pickle.dump(stim_evoked_spike_rates, f)

    with open(pkl_path, "rb") as f:
        stim_evoked_spike_rates = pickle.load(f)
    stim_spike_rates_list.append(stim_evoked_spike_rates)


# %% Dimensionality reduction
variance_threshold = 0.85
pca = PCA(n_components=3)
# Fit PCA on baseline spike rates and transform the data
pca.fit(baseline_spike_rates)  # Learn the structure of baseline data
baseline_pca = pca.transform(baseline_spike_rates)  # Transform baseline data
baseline_num_components = helpers.get_dimensionality(
    baseline_spike_rates, variance_threshold
)

stim_projected_list = []
stim_num_components_list = []

for i, stim_spike_rates in enumerate(stim_spike_rates_list):
    # Project stimulus spike rates onto the PCA components derived from baseline
    stim_projected = pca.transform(stim_spike_rates)
    stim_projected_list.append(stim_projected)

    num_components = helpers.get_dimensionality(stim_spike_rates, variance_threshold)

    stim_num_components_list.append(num_components)

# %% # %% Visualization
views = [(20, -60), (50, 85)]

# Plot and save PCA projections for each view
overlap_list = helpers.plot_and_save_projections(
    baseline_pca,
    stim_projected_list,
    sim_dict["data_path"],
    "pca_projection",
    views=views,
    # xlim=[-10, 10], ylim=[-5, 3], zlim=[-5, 5]
)

stim_channels = [1, 2, 4, 8, 16, 32]  # Number of stimulation channels
plt.figure(figsize=(8, 5))
plt.plot(stim_channels, overlap_list, marker="o")
plt.xticks(stim_channels, labels=stim_channels)
# Add labels and title
plt.xlabel("Number of Stimulated Channels")
plt.ylabel("Volume overlap")
plt.title(
    "Volume overlap between baseline and stimulation-evoked firing rates ellipsoids"
)
plt.legend()


plt.figure(figsize=(8, 5))
plt.plot(
    stim_channels, stim_num_components_list, marker="o", label="Evoked Dimensionality"
)
plt.axhline(
    baseline_num_components, color="r", linestyle="--", label="Baseline Dimensionality"
)  # Add baseline reference line

# Set x-axis tick labels
plt.xticks(stim_channels, labels=stim_channels)

# Add labels and title
plt.xlabel("Number of Stimulated Channels")
plt.ylabel("Dimensionality")
plt.title("Stimulus-Evoked vs Baseline Dimensionality")
plt.legend()

# %%
# i = 500
# neuron_old = currents_at_neurons[i].get()
# amp_times_old = neuron_old['amplitude_times']
# amp_old = neuron_old['amplitude_values']

# neuron_new = current_generators[i].get()
# amp_times_new = neuron_new['amplitude_times']
# amp_new = neuron_new['amplitude_values']

# %% Original code as comparison
# Random stimulation


# stim_spike_rates_list = []
# sim_resolution = sim_dict['sim_resolution']

# for n_groups in N_GROUPS_LIST[2:3]:
#     print(f'n_groups: {n_groups}')
#     nest.ResetKernel()

#     electrodes = RandomStimElectrodes(
#         ch_coordinates, stim_pulse_params, amp_decay_func, sim_resolution)

#     sim_dict["data_path"] = os.path.join(
#         os.getcwd(), f"data20241011/data_randstim_{n_groups}groups_biphasic/"
#     )
#     if os.path.exists(sim_dict["data_path"]):
#         shutil.rmtree(sim_dict["data_path"])

#     network = Network(sim_dict, net_dict, stim_dict)
#     network.create()
#     network.connect()

#     old_stimulations = electrodes.generate_random_stimulation(
#         STIM_CHANNELS,
#         STIM_AMPLITUDES,
#         SIM_TIME_MS,
#         stim_rate_hz=STIM_POISSON_RATE_HZ,
#         n_groups=n_groups,
#     )
#     currents_at_neurons = electrodes.get_current_at_locs(
#         network.neuron_locations)
#     network.simulate_current_input(currents_at_neurons, SIM_TIME_MS)

#     RASTER_INTERVAL = [0, 1000]
#     FIRING_RATE_INTERVAL = [0, 1000]
#     network.evaluate(RASTER_INTERVAL, FIRING_RATE_INTERVAL)

#     fig, axes = plt.subplots(2, 1, figsize=(10, 8))
#     utils.plot_raster(network, time_range_ms=RASTER_INTERVAL, ax=axes[0])
#     electrodes.plot_stim_raster(ax=axes[1], time_range_ms=RASTER_INTERVAL)
#     plt.tight_layout()
#     plt.show()

# neuron0 = currents_at_neurons[1].get()
# amp_times = neuron0['amplitude_times']
# amp = neuron0['amplitude_values']

# Y_old = []
# for i in range(np.sum(network.num_neurons)):
#     neuron_old = currents_at_neurons[i].get()
#     amp_times_old = neuron_old['amplitude_times']
#     amp_old = neuron_old['amplitude_values']
#     Y_old.append(amp_old)

# Y_old = np.array(Y_old)
