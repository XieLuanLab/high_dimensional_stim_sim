import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    hist, bins = np.histogram(all_spikes, bins=np.arange(hist_range[0], hist_range[1] + bin_size, bin_size))
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
            spike_train = spike_train[(spike_train >= time_range_ms[0]) & (spike_train <= time_range_ms[1])]
        
        ax.eventplot(spike_train, lineoffsets=i, colors="k")  # Use ax.eventplot instead of plt.eventplot
    
    # Set axis limits and labels
    if time_range_ms is not None:
        ax.set_xlim(time_range_ms[0], time_range_ms[1])  # Set x-axis to the provided time range
    else:
        ax.set_xlim(0, nn.simulation_time_ms)  # Default to full simulation time
    
    ax.set_ylim(0, nn.n_neurons+1)
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
        smoothed_rate, bin_centers = compute_smoothed_firing_rate(spike_trains[neuron_idx], sigma=50)
        plt.plot(bin_centers, smoothed_rate, label=f"Neuron {neuron_idx}")
        
    if time_range_ms is not None:
        ax.set_xlim(time_range_ms[0], time_range_ms[1])  
        
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Smoothed Firing Rates of Top {plot_top_n} Neurons")
    plt.legend()
    plt.show()
            