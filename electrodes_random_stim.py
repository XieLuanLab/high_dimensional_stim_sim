import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np

UA2PA = 1e6


class RandomStimElectrodes:
    def __init__(
        self,
        ch_coordinates: np.ndarray,
        stim_pulse_params: dict,
        current_disperse_func: callable,
    ):
        """
        ch_coordinates : (n_chs, 2) ndarray of channel coordinates
        current_disperse_func : function(amp_uA, distance_um) -> amp_pA at distance; should support broadcasting
        """
        self.n_chs = len(ch_coordinates)
        self.ch_coordinates = ch_coordinates
        self.current_disperse_func = current_disperse_func
        self.stim_pulse_params = stim_pulse_params
        self.stimulations = None
        self.stim_onset_times_by_ch = {}

    def generate_biphasic_pulse(self, t0, amplitude):
        """
        Generate biphasic pulse timing and amplitude values.
        """
        pulse_width_ms = self.stim_pulse_params.get("pulse_width_ms", 0.2)
        ipi_ms = self.stim_pulse_params.get("ipi_ms", 0.2)

        t1 = np.round(t0 + pulse_width_ms, 1)  # End of cathodic phase
        t2 = np.round(t1 + ipi_ms, 1)  # Start of anodic phase
        t3 = np.round(t2 + pulse_width_ms, 1)  # End of anodic phase

        # Amplitudes corresponding to those times: [-A, 0, A, 0]
        return [t0, t1, t2, t3], [-amplitude * UA2PA, 0, amplitude * UA2PA, 0]

    def generate_random_stimulation(
        self,
        channels,
        amplitude_range,
        duration_ms,
        stim_rate_hz=10,
        min_pulse_spacing=1,
        n_groups=1,
    ):
        """
        Generate random stimulation across channels with randomized amplitudes and
        Poisson-distributed stimulation times.

        Parameters
        ----------
        channels : list
            List of available channels.
        amplitude_range : tuple (min_amp, max_amp)
            Range of amplitudes to randomly assign (in µA).
        duration_ms : float
            Duration of the stimulation (in milliseconds).
        stim_rate_hz : float
            Rate of the Poisson process that determines the average number of stimulations per second.
        n_groups : int
            Number of groups. Each group shares the same stimulation times and amplitudes.

        Returns
        -------
        stimulations : list of dicts
            List of stimulations, where each stimulation is a dict with:
            - 'channel' : The stimulated channel
            - 'times' : List of pulse times (ms)
            - 'amplitudes' : List of amplitudes (µA)
        """

        stimulations = []
        num_channels = len(channels)

        # If number of groups is more than the number of channels, limit it to the number of channels
        n_groups = min(n_groups, num_channels)

        # Divide channels into groups
        channels_per_group = np.array_split(channels, n_groups)

        # Convert rate in Hz to milliseconds (1 second = 1000 ms)
        lambda_poisson = stim_rate_hz / 1000  # Rate in events/ms

        for group in channels_per_group:
            # Generate random stimulation times using Poisson process for the group
            stim_times = []
            current_time = max(
                min_pulse_spacing, np.random.exponential(1 / lambda_poisson)
            )  # First event (NEST requires it to be a strictly positive time)
            while current_time < duration_ms:
                stim_times.append(
                    round(current_time / min_pulse_spacing) * min_pulse_spacing
                )
                current_time += max(
                    min_pulse_spacing, np.random.exponential(1 / lambda_poisson)
                )  # Time to next event

            # Assign a random amplitude for each stimulation time, shared within the group
            self.stim_onset_times_by_ch[
                group[0]
            ] = []  # Initialize for the first channel in the group

            stim_times_and_amplitudes = []
            for stim_time in stim_times:
                amplitude = np.random.randint(*amplitude_range)
                times, amps = self.generate_biphasic_pulse(stim_time, amplitude)
                stim_times_and_amplitudes.append((times, amps))
                self.stim_onset_times_by_ch[group[0]].append(
                    times[0]
                )  # Store onset time for the first channel

            # Apply the same stimulation to all channels in the group
            for ch in group:
                self.stim_onset_times_by_ch[ch] = self.stim_onset_times_by_ch[
                    group[0]
                ].copy()  # Copy times to other channels
                stimulations.append(
                    {
                        "channel": ch,
                        "pulse_params": stim_times_and_amplitudes.copy(),  # Copy the pulse parameters
                    }
                )

        self.stimulations = stimulations

        return stimulations

    def get_current_at_locs(self, locs):
        """
        Get the current at locations over time for each neuron, based on stimulation pulses.

        Parameters
        ----------
        locs : (n_locs, 2) ndarray
            Array of neuron locations.

        Returns
        -------
        current_generators : list of NEST step_current_generator objects
            List of current generators for each neuron, containing the current applied at each timestep.
        """
        if self.stimulations is None:
            print("Stimulations property is empty.")
            return []

        n_locs = locs.shape[0]

        # Compute distances between channels and neuron locations
        distx_ch2loc = np.subtract.outer(
            self.ch_coordinates[:, 0], locs[:, 0]
        )  # (n_chs, n_locs)
        disty_ch2loc = np.subtract.outer(
            self.ch_coordinates[:, 1], locs[:, 1]
        )  # (n_chs, n_locs)
        distance_ch2loc = np.sqrt(
            distx_ch2loc**2 + disty_ch2loc**2
        )  # (n_chs, n_locs)

        # Compute the current contribution for each channel to every neuron location
        currents_ch2loc = []
        for ch_idx, stim in enumerate(self.stimulations):
            ch_current_vals = np.array(
                [amp for _, amps in stim["pulse_params"] for amp in amps]
            )  # All amplitudes
            distances_to_locs = distance_ch2loc[ch_idx, :]  # Distances for this channel

            # Apply the decay function based on distance
            currents_to_locs = self.current_disperse_func(
                ch_current_vals[:, None], distances_to_locs[None, :]
            ).T  # (n_locs, n_changepoints)
            currents_ch2loc.append(currents_to_locs)

        # Now generate the NEST step_current_generator objects for each neuron
        current_generators = []

        for i_loc in range(n_locs):  # For each neuron
            current_changepoints_all = []

            # For each channel, get the times and currents affecting this neuron
            for ch_idx, stim in enumerate(self.stimulations):
                current_times_ch = np.array(
                    [time for times, _ in stim["pulse_params"] for time in times]
                )
                current_vals_ch = currents_ch2loc[ch_idx][
                    i_loc, :
                ]  # Get current values at this location

                # Store the times and corresponding current values
                current_changepoints_all.append(
                    np.column_stack((current_times_ch, current_vals_ch))
                )

            # Stack all changepoints from different channels
            current_changepoints_all = np.vstack(current_changepoints_all)

            # Round times to the nearest simulation resolution
            times_rounded = (
                np.round(current_changepoints_all[:, 0] / nest.resolution)
                * nest.resolution
            )

            # Find unique times and sum the corresponding currents
            unique_times, inverse_indices = np.unique(
                times_rounded, return_inverse=True
            )
            current_vals_reduced = np.zeros_like(unique_times)
            np.add.at(
                current_vals_reduced, inverse_indices, current_changepoints_all[:, 1]
            )

            # Create the NEST step_current_generator for this location
            gen = nest.Create(
                "step_current_generator",
                params=dict(
                    label=f"current_delivery_at_loc_{i_loc}",
                    amplitude_times=unique_times,
                    amplitude_values=current_vals_reduced,
                ),
            )
            current_generators.append(gen)

        return current_generators

    def plot_stim_raster(self, time_range_ms=None, ax=None):
        stim_channel_indices = list(self.stim_onset_times_by_ch.keys())
        num_stim_channels = len(stim_channel_indices)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))
        for i, ch_idx in enumerate(stim_channel_indices):
            stim_train = self.stim_onset_times_by_ch[ch_idx]
            ax.eventplot(
                stim_train, lineoffsets=i + 1, linelengths=0.5, colors="k"
            )  # Plot stim trains

        if time_range_ms is not None:
            ax.set_xlim(
                time_range_ms[0], time_range_ms[1]
            )  # Set x-axis to the provided time range

        ax.set_ylim(0, num_stim_channels + 1)
        ax.set_yticks(range(1, num_stim_channels + 1))
        ax.set_yticklabels([str(ch_idx) for ch_idx in stim_channel_indices])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Stim channel index")
