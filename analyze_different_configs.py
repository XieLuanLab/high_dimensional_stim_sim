import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import helpers
from corcol_params.sim_params import sim_dict

# Analyze results for different stimulation configurations
stim_currents = [0.5, 1, 1.5, 2, 2.5]
N_GROUPS_LIST = [1, 2, 4, 8, 16, 32]  # number of stim electrode groups

results_df = pd.DataFrame(
    columns=["current", "condition", "n_groups", "num_components", "overlap"]
)

for current in stim_currents:
    base_path = os.path.join(
        os.getcwd(), "outputs", f"data_8Hz_k10_scale005_[{current}]uA"
    )

    # Baseline
    sim_dict["data_path"] = os.path.join(base_path, "data_baseline")
    pkl_path = os.path.join(sim_dict["data_path"], "baseline_spike_rates.pkl")
    with open(pkl_path, "rb") as f:
        baseline_spike_rates = pickle.load(f)

    pca = PCA(n_components=3)
    pca.fit(baseline_spike_rates)
    baseline_pca = pca.transform(baseline_spike_rates)
    baseline_num_components = helpers.get_dimensionality(
        baseline_spike_rates, variance_threshold=0.85
    )

    baseline_row = pd.DataFrame(
        {
            "current": [current],
            "condition": ["baseline"],
            "n_groups": [None],
            "num_components": [baseline_num_components],
            "overlap": [None],
        }
    )
    results_df = pd.concat([results_df, baseline_row], ignore_index=True)

    # Iterate through stim channel groups
    stim_projected_list = []
    stim_num_components_list = []

    for n_groups in N_GROUPS_LIST:
        sim_dict["data_path"] = os.path.join(
            base_path, f"data_randstim_{n_groups}groups/"
        )
        pkl_path = os.path.join(
            sim_dict["data_path"], f"{n_groups}groups_stim_spike_rates.pkl"
        )
        pkl_path_stim_pulses = os.path.join(
            sim_dict["data_path"], f"{n_groups}groups_stim_pulses.pkl"
        )
        with open(pkl_path, "rb") as f:
            stim_evoked_spike_rates = pickle.load(f)

        stim_projected = pca.transform(stim_evoked_spike_rates)
        stim_projected_list.append(stim_projected)

        num_components = helpers.get_dimensionality(
            stim_evoked_spike_rates, variance_threshold=0.85
        )
        stim_num_components_list.append(num_components)

    overlap_list, _, _ = helpers.compute_all_overlaps(baseline_pca, stim_projected_list)

    stim_data = pd.DataFrame(
        {
            "current": [current] * len(N_GROUPS_LIST),
            "condition": ["stim"] * len(N_GROUPS_LIST),
            "n_groups": N_GROUPS_LIST,
            "num_components": stim_num_components_list,
            "overlap": overlap_list,
        }
    )

    results_df = pd.concat([results_df, stim_data], ignore_index=True)

# %% Plot the results for 32 channel group for all currents

df_32 = results_df[results_df["n_groups"] == 32]
plt.plot(np.arange(5), df_32["overlap"])
plt.xticks(np.arange(5), np.arange(0.5, 3, 0.5))
plt.xlabel("Stimulation Current (uA)")
plt.ylabel("Overlap Score (Jaccard Index)")
plt.title("Overlap Comparison for 32 Stim Groups at Different Currents")

# %% Plot all overlaps for different currents

n_rows = len(stim_currents) // 2 + len(stim_currents) % 2
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 6), constrained_layout=True)
axes = axes.flatten()

# Loop through each stimulation current and create a subplot
for i, current in enumerate(stim_currents):
    df_current = results_df[results_df["current"] == current]

    df_current = df_current.dropna(subset=["n_groups", "overlap"])
    df_current["n_groups"] = pd.to_numeric(df_current["n_groups"], errors="coerce")
    df_current["overlap"] = pd.to_numeric(df_current["overlap"], errors="coerce")

    n_groups = df_current["n_groups"].values
    overlap = df_current["overlap"].values

    if len(n_groups) > 0 and len(overlap) > 0:
        ax = axes[i]
        ax.plot(range(6), overlap, marker="o", linestyle="-", color="C0")

        ax.set_xticks(range(6))
        ax.set_xticklabels(n_groups, rotation=45)

        ax.set_title(f"Current = {current} µA")
        ax.set_xlabel("N Groups")
        ax.set_ylabel("Overlap")

if len(stim_currents) % 2 != 0:
    fig.delaxes(axes[-1])

plt.suptitle("Overlap vs N Groups for Different Currents", fontsize=16)
plt.tight_layout()
