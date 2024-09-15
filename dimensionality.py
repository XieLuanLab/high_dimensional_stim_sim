import numpy as np
import nest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from network import Network
import utils as utils

def compute_smoothed_firing_rate(spike_train, sim_time_ms, sigma=50, time_resolution=1):
    """
    Compute a smoothed firing rate using Gaussian kernel convolution.
    """
    
    time_bins = np.arange(0, sim_time_ms, time_resolution)
    spike_counts, _ = np.histogram(spike_train, bins=time_bins)

    # Gaussian smoothing kernel
    kernel_width = int(3 * sigma / time_resolution)
    kernel = np.exp(-np.linspace(-kernel_width, kernel_width, 2 * kernel_width + 1)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)

    # Smooth the spike counts
    smoothed_rate = np.convolve(spike_counts, kernel, mode='same')
    return smoothed_rate, time_bins[:-1]


def get_dimensionality(spike_trains, sim_time_ms=30000, window_ms=100, overlap_ms=50, plot_scree=True):
    step_ms = window_ms - overlap_ms  # Step size for overlapping windows
    num_windows = int(np.floor((sim_time_ms - window_ms) / step_ms)) + 1  # Adjust number of windows based on step size
    
    num_neurons = len(spike_trains)
    windows = np.linspace(0, sim_time_ms - window_ms, num_windows)  # Start of each window
    spike_rates = np.zeros((num_windows, num_neurons))
    
    # Calculate firing rates for each window
    for i in range(num_neurons):
        smoothed_rate, _ = compute_smoothed_firing_rate(spike_trains[i], sim_time_ms, sigma=20)
        for j in range(num_windows):
            start_time = windows[j]
            end_time = start_time + window_ms
            # num_spikes = np.sum((spike_trains[i] > start_time) & (spike_trains[i] <= end_time))
            # spike_rates[j, i] = (num_spikes / window_ms)
            spike_rates[j, i] = np.mean(smoothed_rate[int(start_time):int(end_time)])
    
    # Standardize the spike rates
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(spike_rates)
    
    # Perform PCA
    pca = PCA()
    pca.fit(data_standardized)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(explained_variance >= 0.95) + 1
    print(f'Number of components explaining 95% variance: {num_components}')
    
    # Plot Scree plot if needed
    if plot_scree:
        plt.figure()
        plt.plot(explained_variance, label="Cumulative Explained Variance")
        plt.axvline(num_components, color='k', linestyle='--', label=f"{num_components} components")
        
        # Annotate the number of components
        plt.text(num_components + 1, 0.9, f'{num_components} components', 
                 verticalalignment='center', color='k')
        
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        plt.grid(True)
        plt.show()
    
    return spike_rates
