import os
import pickle
import shutil

import matplotlib.pyplot as plt
import nest
import numpy as np
from sklearn.decomposition import PCA

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
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
RANDOM_STIM = True  # else, deterministic

PRESIM_TIME_MS = sim_dict["t_presim"]
SIM_TIME_MS = sim_dict["t_sim"]
WINDOW_MS = 500
OVERLAP_MS = 400
RASTER_PLOT_TIME_MS = 500
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

Nscale = net_dict['N_scaling'] # assume K and N same 
sim_time_s = int(sim_dict['t_sim'] / 1000)
stim_amps_str = STIM_AMPLITUDES[0] if len(STIM_AMPLITUDES) > 0 else STIM_AMPLITUDES
base_path = os.path.join(os.getcwd(), "outputs", 
                         f"data_{STIM_POISSON_RATE_HZ}Hz_{Nscale}scale_{stim_amps_str}uA_{sim_time_s}s")

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
    baseline_spike_rates = helpers.compute_spike_rates(
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

stim_spike_rates_list = []

for n_groups in N_GROUPS_LIST:
    sim_dict["data_path"] = os.path.join(base_path, f"data_randstim_{n_groups}groups/")
    pkl_path = os.path.join(
        sim_dict["data_path"], f"{n_groups}groups_stim_spike_rates.pkl"
    )
    pkl_path_stim_pulses = os.path.join(
        sim_dict["data_path"], f"{n_groups}groups_stim_pulses.pkl"
    )
    # Check if spike rates saved previously and if so, skip processing
    if not os.path.exists(pkl_path):
        print(f"\n\n***** n_groups: {n_groups} *****\n\n")
        nest.ResetKernel()

        electrodes = StimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)

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

        current_generators = electrodes.get_current_generators()

        network.simulate_current_input(
            current_generators, time_ms=SIM_TIME_MS
        )  # stimulate and record

        # Get evoked spike rates and append to list
        stim_evoked_spike_trains = network.get_spike_train_list()
        stim_evoked_spike_rates = helpers.compute_spike_rates(
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

        with open(pkl_path_stim_pulses, "wb") as f:
            pickle.dump(electrodes.stim_onset_times_by_ch, f)

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
views = [(12, -36), (15, -10)]
views = [(10, -100)]

# Plot and save PCA projections for each view
helpers.plot_projections(
    baseline_pca,
    stim_projected_list,
    sim_dict["data_path"],
    "pca_projection",
    views=views,
    xlim=[-0.08, 0.08],
    ylim=[0.07, -0.2],
    zlim=[-0.05, 0.04],
)

overlap_list, _, _ = helpers.compute_all_overlaps(baseline_pca, stim_projected_list)

plt.suptitle(f"{stim_amps_str} uA")

plt.savefig(
    os.path.join(base_path, f"pca_projection_ellipsoids.png")
)
plt.close()
# %%
stim_channels = [1, 2, 4, 8, 16, 32]  # Number of stimulation channels
plt.figure(figsize=(8, 5))
plt.plot(np.arange(6), overlap_list, marker="o")
plt.xticks(np.arange(6), labels=stim_channels)
plt.xlabel("Number of Stimulated Channels")
plt.ylabel("Jaccard Index")
plt.title("Volume Overlap")
plt.legend().set_visible(False)
plt.savefig(
    os.path.join(base_path, f"volume overlap vs num ch.png")
)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(6), stim_num_components_list, marker="o", label="Evoked Dimensionality"
)
plt.axhline(
    baseline_num_components, color="r", linestyle="--", label="Baseline Dimensionality"
)  # Add baseline reference line

plt.xticks(np.arange(6), labels=stim_channels)

plt.xlabel("Number of Stimulated Channels")
plt.ylabel("Dimensionality")
plt.title("Stimulus-Evoked vs Baseline Dimensionality")
plt.legend()
plt.savefig(
    os.path.join(base_path, f"dimensionality vs num ch.png")
)
plt.close()