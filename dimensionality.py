import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import utils as utils


def compute_smoothed_firing_rate(spike_train, sim_time_ms, sigma=50, time_resolution=1):
    """
    Compute smoothed firing rate for spike train using Gaussian kernel.

    Parameters
    ----------
    spike_train : np.array
        Array of spike times in milliseconds for a single neuron.
    sim_time_ms : int
        Total simulation time in milliseconds.
    sigma : int, optional
        Standard deviation of the Gaussian kernel for smoothing the firing rate. The default is 50.
    time_resolution : int, optional
        Resolution of time bins in millisecond. The default is 1.

    Returns
    -------
    smoothed_rate : np.array
        Array of the smoothed firing rate over time.
    time_bins : np.array
        Array of time bin centers used for calculating the firing rate.

    """

    time_bins = np.arange(0, sim_time_ms, time_resolution)
    spike_counts, _ = np.histogram(spike_train, bins=time_bins)

    # Gaussian smoothing kernel
    kernel_width = int(3 * sigma / time_resolution)
    kernel = np.exp(
        -np.linspace(-kernel_width, kernel_width, 2 * kernel_width + 1) ** 2
        / (2 * sigma**2)
    )
    kernel /= np.sum(kernel)

    # Smooth the spike counts
    smoothed_rate = np.convolve(spike_counts, kernel, mode="same")
    return smoothed_rate, time_bins[:-1]


def compute_spike_rates(
    spike_trains, sim_time_ms, window_ms, overlap_ms, presim_time_ms=0, sigma=20
):
    """
    Compute smoothed spike rates for each neuron over overlapping time windows.

    Parameters
    ----------
    spike_trains : list of np.array
        List where each element is an array of spike times in milliseconds for a single neuron.
    sim_time_ms : int
        Total simulation time in milliseconds.
    window_ms : int
        Duration of each window in ms to calculate spike rates.
    overlap_ms : int
        Overlap duration between consecutive windows in milliseconds.
    presim_time_ms: int
        Pre-simulation time to subtract from spike trains.
    sigma : float, optional
        Standard deviation of the Gaussian kernel for smoothing the firing rate. The default is 20.

    Returns
    -------
    spike_rates : np.ndarray
        2D array with shape (num_windows, num_neurons) representing smoothed spike rates.
        Rows represent different time windows, and columns represent different neurons.

    """
    step_ms = window_ms - overlap_ms
    num_windows = int(np.floor((sim_time_ms - window_ms) / step_ms)) + 1
    num_neurons = len(spike_trains)
    windows = np.linspace(0, sim_time_ms - window_ms, num_windows)
    spike_rates = np.zeros((num_windows, num_neurons))

    # Adjust for pre-simulation time
    if presim_time_ms > 0:
        spike_trains = [
            np.array(spike_train) - presim_time_ms for spike_train in spike_trains
        ]

    for i, spike_train in enumerate(spike_trains):
        smoothed_rate, _ = compute_smoothed_firing_rate(
            spike_train, sim_time_ms, sigma=sigma
        )
        for j, start_time in enumerate(windows):
            end_time = start_time + window_ms
            spike_rates[j, i] = np.mean(smoothed_rate[int(start_time) : int(end_time)])

    return spike_rates


def get_dimensionality(spike_rates, plot_scree=True):
    """
    Compute the number of principal components required to explain 95% of the variance in spike rates.

    Parameters
    ----------
    spike_rates : np.ndarray
        2D array where rows represent different time windows and columns represent neurons.
    plot_scree : bool, optional
        If True, plot a scree plot showing cumulative explained variance. Default is True.

    Returns
    -------
    num_components : int
        Number of PCA components explaining 95% of the variance.

    """
    pca = PCA()
    pca.fit(spike_rates)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(explained_variance >= 0.95) + 1
    print(f"Number of components explaining 95% variance: {num_components}")

    if plot_scree:
        plt.figure()
        plt.plot(explained_variance, label="Cumulative Explained Variance")
        plt.axvline(
            num_components,
            color="k",
            linestyle="--",
            label=f"{num_components} components",
        )
        # Annotate the number of components explaining 95% variance
        plt.text(
            num_components + 1,
            0.95,
            f"{num_components} components",
            verticalalignment="center",
            color="k",
        )

        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.show()

    return num_components


# TODO: don't know if functions necessary


def fit_and_transform_baseline_pca(baseline_spike_rates):
    """
    Fit PCA on baseline spike rates and transform the data.

    Parameters:
        baseline_spike_rates (ndarray): Baseline spike rates (time steps x neurons).

    Returns:
        baseline_pca (ndarray): Transformed baseline data in PCA space.
        pca (PCA): The fitted PCA object.
    """
    pca = PCA(n_components=3)
    baseline_pca = pca.fit_transform(baseline_spike_rates)

    return baseline_pca, pca


def project_stim_to_pca(stim_spike_rates, pca):
    """
    Project stimulus spike rates onto the PCA components derived from baseline.

    Parameters:
        stim_spike_rates (ndarray): Stimulus spike rates (time steps x neurons).
        pca (PCA): The fitted PCA object from the baseline spike rates.

    Returns:
        stim_projected (ndarray): Stimulus data projected onto baseline PCA components.
    """
    # Project the stim data onto the baseline PCA components
    stim_projected = pca.transform(stim_spike_rates)

    return stim_projected


def plot_trajectories(
    baseline_pca, sigma=2, stim_projected=None, stim_color=None, ax=None
):
    """
    Plots the baseline PCA and stimulus projections on the same 3D axis with Gaussian smoothing.

    Parameters:
        baseline_pca (ndarray): Transformed baseline data projected onto top 3 PCA components.
        sigma (float): Standard deviation for Gaussian kernel to smooth the trajectories.
        stim_projected (ndarray): Stimulus data projected onto the PCA components.
        stim_color (str): Color for the stimulus trajectory.
        ax (matplotlib axis): Existing 3D axis to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Apply Gaussian filter to baseline PCA components
    baseline_smoothed = np.empty_like(baseline_pca)
    for i in range(3):  # Loop over each PCA component (x, y, z)
        baseline_smoothed[:, i] = gaussian_filter1d(baseline_pca[:, i], sigma=sigma)

    # Plot the smoothed baseline trajectory
    ax.plot(
        baseline_smoothed[:, 0],
        baseline_smoothed[:, 1],
        baseline_smoothed[:, 2],
        color="k",
        linewidth=0.5,
        alpha=0.7,
        label="Baseline Smoothed",
    )

    if stim_projected is not None:
        # Apply Gaussian filter to stimulus components
        stim_smoothed = np.empty_like(stim_projected)
        for i in range(3):
            stim_smoothed[:, i] = gaussian_filter1d(stim_projected[:, i], sigma=sigma)

        # Plot the smoothed stimulus trajectory
        ax.plot(
            stim_smoothed[:, 0],
            stim_smoothed[:, 1],
            stim_smoothed[:, 2],
            color=stim_color,
            linewidth=0.5,
            alpha=0.7,
            label="Stimulus Smoothed",
        )

    ax.set_title("3D Projection with Smoothed Trajectories")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    return ax


def plot_and_save_projections(
    baseline,
    stim_projected_list,
    projection_type,
    output_dir,
    file_name,
    views,
    n_groups_list,
    xlim=None,
    ylim=None,
    zlim=None,
):
    baseline_gmm = GaussianMixture(n_components=1, covariance_type="full")
    baseline_gmm.fit(baseline)
    baseline_mean = baseline_gmm.means_[0]
    baseline_cov = baseline_gmm.covariances_[0]

    volume_overlap_list = []

    # Loop over the views to generate a separate figure for each view
    for view_index, view in enumerate(views):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

        for i, stim_projected in enumerate(stim_projected_list):
            ax = fig.add_subplot(gs[i], projection="3d")

            # Set axis limits
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if zlim:
                ax.set_zlim(zlim)

            # Plot trajectories
            plot_trajectories(baseline, stim_projected, stim_color="C0", ax=ax)

            # Fit Gaussian Mixture Model for stimulus data
            stim_gmm = GaussianMixture(n_components=1, covariance_type="full")
            stim_gmm.fit(stim_projected)
            stim_mean = stim_gmm.means_[0]
            stim_cov = stim_gmm.covariances_[0]

            # Plot ellipsoids
            utils.plot_gaussian_ellipsoid(
                stim_mean,
                stim_cov,
                ax=ax,
                n_std=2,
                color="k",
                alpha=0.2,
                wireframe=True,
            )
            utils.plot_gaussian_ellipsoid(
                baseline_mean,
                baseline_cov,
                ax=ax,
                n_std=2,
                color="k",
                alpha=0.2,
                wireframe=True,
            )

            # Set view and title
            ax.view_init(elev=view[0], azim=view[1])
            ax.set_title(f"{n_groups_list[i]} stim channels", fontsize=12)

            # Calculate and store radii difference
            if view_index == 0:
                volume_overlap = utils.compare_overlap_volume(baseline_cov, stim_cov)
                volume_overlap_list.append(volume_overlap)

        plt.tight_layout()
        plt.suptitle(
            f"{projection_type} Projection (View {view_index + 1})", fontsize=16
        )

        # Save the figure for the current view
        file_path = os.path.join(output_dir, f"{file_name}_view_{view_index + 1}.png")
        plt.savefig(file_path)
        plt.show()

    return volume_overlap_list
