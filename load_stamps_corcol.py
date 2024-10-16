from itertools import groupby

import numpy as np
import tqdm

from helpers import __load_spike_times


def load_stamps_corcol(data_dir):
    """
    Load spike stamps from a directory

    Parameters
    ----------
    data_dir : str
        Directory containing the spike stamps (.dat files).

    Returns
    -------
    stamps : dict
        (neuron id, spike stamps) pairs.
    """
    sd_names, node_ids, data = __load_spike_times(data_dir, "spike_recorder", 0, np.inf)
    last_node_id = node_ids[-1, -1]
    all_neuron_stamps = {}
    for i_pop, sd_name in enumerate(sd_names):
        times = data[i_pop]["time_ms"]
        neurons = np.abs(data[i_pop]["sender"] - last_node_id) + 1
        for neuron_id, spike_indices in groupby(
            sorted(range(len(neurons)), key=neurons.__getitem__),
            key=neurons.__getitem__,
        ):
            # one spike stamp file per neuron
            stamp = times[list(spike_indices)]
            # neuron_name = "neuron%d_%s"%(neuron_id, sd_name)
            # assert neuron_name not in all_neuron_stamps
            # all_neuron_stamps[neuron_name] = stamp
            all_neuron_stamps[neuron_id] = stamp
    return all_neuron_stamps


if __name__ == "__main__":
    from time import time

    npz_path = "./data20241011/data_randstim_32groups/"
    ts = time()
    stamps = load_stamps_corcol(npz_path)
    print("Loaded %d stamps in %.1f sec." % (len(stamps), time() - ts))
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    plt.figure(figsize=(8, 8))
    for k, v in stamps.items():
        plt.plot(v, np.ones_like(v) * k, "k.", markersize=0.1)
    # plt.show()
    plt.savefig("tmp_stamps_corcol.png")
    plt.close()
