import os
import pickle
import shutil

import matplotlib.pyplot as plt
import nest
import numpy as np

import helpers
from corcol_params.network_params import net_dict
from corcol_params.sim_params import sim_dict
from corcol_params.stimulus_params import stim_dict
from electrodes_stim import StimElectrodes
from network_cortcol import Network

# ====================PROBE CONFIGURATION======================================
volume_v_min = 200  # um
ch_vcoords = np.arange(32)[::-1] * 60  # index 0 -> 31 is deep -> shallow
ch_hcoords = np.zeros(32)
ch_coordinates = np.stack([ch_hcoords, ch_vcoords], axis=1)

sigma_e_um = 2.76e-7
conductivity_constant = 10
STIM_AMPLITUDES = [2]  # uA

INTERPULSE_INTERVAL_MS = 10
INTERPATTERN_INTERVAL_MS = 1000
PRESIM_TIME_MS = sim_dict["t_presim"]
SIM_TIME_MS = sim_dict["t_sim"]
WINDOW_MS = 500
OVERLAP_MS = 400
RASTER_PLOT_TIME_MS = 500
RASTER_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + RASTER_PLOT_TIME_MS]
FIRING_RATE_INTERVAL = [PRESIM_TIME_MS, PRESIM_TIME_MS + SIM_TIME_MS]
stim_pulse_params = {"pulse_width_ms": 0.2, "ipi_ms": 0.2}


def amp_decay_func(amp_uA, dist_um):
    """Calculate decayed amplitude of induced current."""
    return (
        amp_uA
        * 1e-6
        * conductivity_constant
        / (4 * np.pi * sigma_e_um * (dist_um + 20))
    )


def run_baseline_simulation(network, base_path):
    """Run baseline simulation and save results."""

    network.simulate_baseline(PRESIM_TIME_MS)
    network.simulate_baseline(SIM_TIME_MS)

    baseline_spike_trains = network.get_spike_train_list()
    baseline_spike_rates = helpers.compute_spike_rates(
        baseline_spike_trains,
        SIM_TIME_MS,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )

    # Save baseline results
    pkl_path = os.path.join(base_path, "baseline_spike_rates.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(baseline_spike_rates, f)

    # Plot baseline raster
    fig, ax = plt.subplots(figsize=(8, 4))
    network.evaluate(
        RASTER_INTERVAL, FIRING_RATE_INTERVAL, title="Baseline Activity", raster_ax=ax
    )
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "baseline_raster.png"))
    plt.close(fig)

    return baseline_spike_rates


def run_stimulation_simulation(network, electrodes, pattern_id, pattern, stim_path):
    """Run stimulation with deterministic pattern and save results."""
    electrodes.generate_deterministic_stimulation(
        pattern["channels"],
        pattern["times"],
        STIM_AMPLITUDES,
        SIM_TIME_MS,
        INTERPATTERN_INTERVAL_MS,
    )

    electrodes.compute_impulse_response_matrix(network.neuron_locations)
    electrodes.compute_stim_current_matrix()
    electrodes.calculate_induced_current_matrix()

    current_generators = electrodes.get_current_generators(
        presim_time_ms=PRESIM_TIME_MS
    )
    network.simulate_current_input(current_generators, time_ms=SIM_TIME_MS)

    stim_evoked_spike_trains = network.get_spike_train_list()
    stim_evoked_spike_rates = helpers.compute_spike_rates(
        stim_evoked_spike_trains,
        SIM_TIME_MS,
        WINDOW_MS,
        OVERLAP_MS,
        presim_time_ms=PRESIM_TIME_MS,
    )

    # Save stimulation results
    with open(os.path.join(stim_path, f"{pattern_id}_spike_rates.pkl"), "wb") as f:
        pickle.dump(stim_evoked_spike_rates, f)

    # Plot stimulation raster
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    network.evaluate(
        RASTER_INTERVAL,
        FIRING_RATE_INTERVAL,
        title=f"Stimulation Pattern {pattern_id}",
        raster_ax=axes[0],
    )
    electrodes.plot_stim_raster(ax=axes[1], time_range_ms=RASTER_INTERVAL)
    plt.tight_layout()
    plt.savefig(os.path.join(stim_path, f"stim_raster_{pattern_id}.png"))
    plt.close(fig)

    return stim_evoked_spike_rates


# Define stimulation patterns
A_ch = np.arange(32)
A_times = np.arange(1, (32 * INTERPULSE_INTERVAL_MS) + 1, INTERPULSE_INTERVAL_MS)
B_ch = np.arange(32)[::-1]
B_times = np.arange(1, (32 * INTERPULSE_INTERVAL_MS) + 1, INTERPULSE_INTERVAL_MS)

pattern_dict = {
    "A": {"channels": A_ch, "times": A_times},
    "B": {"channels": B_ch, "times": B_times},
}

# Main loop to run simulations
for pattern_id, pattern in pattern_dict.items():
    base_path = os.path.join(os.getcwd(), "outputs", f"deterministic_{pattern_id}")

    # Create the base directory
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    baseline_path = os.path.join(base_path, "baseline")
    os.makedirs(baseline_path, exist_ok=True)

    # Run baseline simulation
    nest.ResetKernel()
    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    baseline_rates = run_baseline_simulation(network, baseline_path)

    # Run stimulation simulation
    stim_path = os.path.join(base_path, "stimulation")
    os.makedirs(stim_path, exist_ok=True)

    nest.ResetKernel()
    network = Network(sim_dict, net_dict, stim_dict)
    network.create()
    network.connect()
    electrodes = StimElectrodes(ch_coordinates, stim_pulse_params, amp_decay_func)
    stim_rates = run_stimulation_simulation(
        network, electrodes, pattern_id, pattern, stim_path
    )
