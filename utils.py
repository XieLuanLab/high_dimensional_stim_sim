import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def sort_and_groupby(arr, key=None):
    """
    Group sorted iterable by key using NumPy for faster performance.

    Parameters:
    arr : array-like
        The input array or list.
    key : function, optional
        The function to extract the key for grouping. If None, groups based on raw values.

    Returns:
    Generator of (unique_key, grouped_elements)
    """
    # Convert input iterable to NumPy array
    arr = np.asarray(arr)

    if key is not None:
        keys = np.array([key(item) for item in arr])
    else:
        keys = arr

    sorted_indices = np.argsort(keys)
    sorted_arr = arr[sorted_indices]
    sorted_keys = keys[sorted_indices]

    unique_keys, group_boundaries = np.unique(sorted_keys, return_index=True)
    grouped_elements = np.split(sorted_arr, group_boundaries[1:])

    # Yield each unique key and its corresponding group of elements
    for unique_key, group in zip(unique_keys, grouped_elements):
        yield unique_key, group


def compute_smoothed_firing_rate(trial_data, bin_size=10, sigma=30):
    """
    Computes smoothed firing rate from spike times using a Gaussian kernel.

    Parameters:
    - trial_data : list of lists of spike times in ms where each nested list is a trial.
    - bin_size : size of time bins in ms.
    - sigma : standard deviation of the Gaussian kernel in ms.

    Returns:
    - smoothed_rate : np.ndarray of smoothed firing rate in Hz (spikes/s).
    - times : np.ndarray of corresponding time points in ms.
    """
    # Flatten the list of lists of spike times
    if len(trial_data == 0):
        trial_data = [trial_data]
    all_spikes = [spike for trial in trial_data for spike in trial]

    if not all_spikes:  # Check if all_spikes is empty
        return np.array([]), np.array([])

    num_trials = len(trial_data)  # Number of trials

    # Adjust the spike time range to account for the Gaussian kernel
    kernel_extent = 3 * sigma
    hist_range = (min(all_spikes) - kernel_extent, max(all_spikes) + kernel_extent)

    # Create a histogram of the spikes
    hist, bins = np.histogram(
        all_spikes, bins=np.arange(hist_range[0], hist_range[1] + bin_size, bin_size)
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Normalize histogram to get average spike count per trial per bin
    hist = hist / num_trials

    # Convert average spike count per bin to firing rate in Hz (spikes/s)
    hist = hist / (bin_size / 1000)  # Convert bin size to seconds and divide

    # Create a Gaussian kernel
    kernel_range = np.arange(-3 * sigma, 3 * sigma + bin_size, bin_size)
    kernel = np.exp(-(kernel_range**2) / (2 * sigma**2))
    kernel /= sum(kernel)

    # Convolve the spike histogram with the Gaussian kernel
    smoothed_rate_full = np.convolve(hist, kernel, mode="full")

    # Extract the 'valid' portion of the convolution output
    valid_start_idx = len(kernel) // 2
    valid_end_idx = valid_start_idx + len(hist)
    smoothed_rate = smoothed_rate_full[valid_start_idx:valid_end_idx]

    return smoothed_rate, bin_centers


def plot_raster(nn, time_range_ms=None, ax=None):
    spike_trains = nn.get_spiketrains()
    neuron_inds_sorted = np.argsort(nn.neuron_locations[:, 1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(nn.n_neurons):
        idx = neuron_inds_sorted[i]
        spike_train = spike_trains[idx]

        # If a time_range is provided, only plot spikes within that range
        if time_range_ms is not None:
            spike_train = spike_train[
                (spike_train >= time_range_ms[0]) & (spike_train <= time_range_ms[1])
            ]

        # Use ax.eventplot instead of plt.eventplot
        ax.eventplot(spike_train, lineoffsets=i, colors="k")

    # Set axis limits and labels
    if time_range_ms is not None:
        # Set x-axis to the provided time range
        ax.set_xlim(time_range_ms[0], time_range_ms[1])
    else:
        # Default to full simulation time
        ax.set_xlim(0, nn.simulation_time_ms)

    ax.set_ylim(0, nn.n_neurons + 1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index (depth-reordered)")


def plot_firing_rates(nn, time_range_ms=None, plot_top_n=5, ax=None):
    spike_trains = nn.get_spiketrains()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    spike_counts = [len(train) for train in spike_trains]
    if plot_top_n == 0:
        top_n_neuron_indices = np.arange(nn.n_neurons)
    else:
        top_n_neuron_indices = np.argsort(spike_counts)[-plot_top_n:]

    for i in range(len(top_n_neuron_indices)):
        neuron_idx = top_n_neuron_indices[i]
        smoothed_rate, bin_centers = compute_smoothed_firing_rate(
            spike_trains[neuron_idx], sigma=50
        )
        plt.plot(bin_centers, smoothed_rate, label=f"Neuron {neuron_idx}")

    if time_range_ms is not None:
        ax.set_xlim(time_range_ms[0], time_range_ms[1])

    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Smoothed Firing Rates of Top {plot_top_n} Neurons")
    plt.legend()
    plt.show()


def plot_gaussian_ellipsoid(
    mean, cov, ax=None, n_std=2, color="k", alpha=0.3, wireframe=True
):
    """
    Plots an ellipsoid representing the Gaussian defined by mean and covariance matrix.

    Parameters:
    - mean: Center of the ellipsoid (mean of Gaussian).
    - cov: Covariance matrix of the Gaussian.
    - ax: Existing 3D axis to plot on. Creates new one if None.
    - n_std: Number of standard deviations to scale the ellipsoid radii. (2 stds includes 95% of points)
    - color: Color of the ellipsoid.
    - alpha: Transparency of the ellipsoid.
    - wireframe: Plot wireframe
    """
    # Create a 3D grid of points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale and rotate the points using the covariance matrix
    sphere_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Decompose the covariance matrix to obtain radii and rotation
    radii, rotation = np.linalg.eigh(cov)
    # Scale the standard deviation by n_std (e.g., 2 for 95% CI)
    radii = n_std * np.sqrt(radii)
    ellipsoid_points = sphere_points @ np.diag(radii) @ rotation.T

    # Translate the points to the mean
    ellipsoid_points += mean

    # Reshape the points for plotting
    x_ellipsoid = ellipsoid_points[:, 0].reshape(x.shape)
    y_ellipsoid = ellipsoid_points[:, 1].reshape(y.shape)
    z_ellipsoid = ellipsoid_points[:, 2].reshape(z.shape)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot the ellipsoid surface
    if wireframe:
        ax.plot_wireframe(
            x_ellipsoid,
            y_ellipsoid,
            z_ellipsoid,
            color=color,
            alpha=alpha,
            rcount=20,
            ccount=20,
            linewidth=1,
        )
    else:
        ax.plot_surface(
            x_ellipsoid,
            y_ellipsoid,
            z_ellipsoid,
            color=color,
            alpha=alpha,
            rstride=4,
            cstride=4,
            linewidth=0,
        )

    return ax


def compute_volume_overlap(mean_1, cov_1, mean_2, cov_2, n_samples=10000):
    """
    Compute an estimate of the volume overlap between two Gaussian distributions.

    Parameters
    ----------
    mean_1 : ndarray
        Mean of the first Gaussian distribution.
    cov_1 : ndarray
        Covariance matrix of the first Gaussian distribution.
    mean_2 : ndarray
        Mean of the second Gaussian distribution.
    cov_2 : ndarray
        Covariance matrix of the second Gaussian distribution.
    n_samples : int, optional
        Number of samples to use for Monte Carlo estimation. Default is 10000.

    Returns
    -------
    overlap_fraction : float
        Estimated fraction of overlapping volume between the two Gaussians.
    samples_1 : ndarray
        Samples drawn from the first Gaussian distribution.
    samples_2 : ndarray
        Samples drawn from the second Gaussian distribution.

    """
    # Generate random samples from both Gaussian distributions
    samples_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples)
    samples_2 = np.random.multivariate_normal(mean_2, cov_2, n_samples)

    # Evaluate the densities of the samples under both Gaussians
    density_1_samples_1 = multivariate_normal.pdf(samples_1, mean=mean_1, cov=cov_1)
    density_2_samples_1 = multivariate_normal.pdf(samples_1, mean=mean_2, cov=cov_2)

    density_1_samples_2 = multivariate_normal.pdf(samples_2, mean=mean_1, cov=cov_1)
    density_2_samples_2 = multivariate_normal.pdf(samples_2, mean=mean_2, cov=cov_2)

    # Estimate the overlap for both sets of samples
    overlap_1 = np.mean(
        np.minimum(density_1_samples_1, density_2_samples_1)
        / np.maximum(density_1_samples_1, density_2_samples_1)
    )
    overlap_2 = np.mean(
        np.minimum(density_1_samples_2, density_2_samples_2)
        / np.maximum(density_1_samples_2, density_2_samples_2)
    )

    # Average the two overlap estimates
    overlap_fraction = (overlap_1 + overlap_2) / 2

    return overlap_fraction, samples_1, samples_2
