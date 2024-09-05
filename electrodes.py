import nest
import numpy as np

from utils import sort_and_groupby

UA2PA = 1e6

class Electrodes:
    def __init__(self, ch_coordinates : np.ndarray, current_disperse_func: callable):
        """
        ch_coordinates : (n_chs, 2) ndarray of channel coordinates
        current_disperse_func : function(amp, distance) -> amp at distance; should support broadcasting
        """
        self.n_chs = len(ch_coordinates)
        self.ch_coordinates = ch_coordinates
        self.current_disperse_func = current_disperse_func
    
    # @staticmethod
    def set_biphasic_pulsetrain(self, stim_ch_params, **kwargs):
        """
        Generate biphasic pulse train currents
        
        Parameters
        ----------
        stim_ch_params : list of dicts that contain the following fields:
            - ch_idx
            - ampl_ua
            - freq_hz
            - npulses
            - ipi_ms
            - pulsewidth_ms
            - onset_time_ms
        kwargs : 
            - ???
        
        Outcome
        -------
        set self.current_changepoints_by_ch : list [n_chs] of current changespoints.
            Each list is a (n_changepoints, 2) where the first column is time 
            and the second is the current amplitude (pA)
        
        """
        # initialize with a 0 current at time 0, so that we don't run into empty list issues later
        current_changepoints_by_ch = [ [[nest.resolution,0]] for _ in range(self.n_chs)] 
        for stim_param in stim_ch_params:
            ch_idx = stim_param['ch_idx']
            ampl_ua = stim_param['ampl_ua']
            freq_hz = stim_param['freq_hz']
            npulses = stim_param['npulses']
            ipi_ms = stim_param['ipi_ms']
            pulsewidth_ms = stim_param['pulsewidth_ms']
            onset_time_ms = stim_param['onset_time_ms']
            if onset_time_ms == 0:
                current_changepoints_by_ch[ch_idx] = []
            for ipulse in range(npulses):
                # theare are 4 change points
                t0 = onset_time_ms + ipulse * 1000/freq_hz # onset cathodic phase
                t1 = t0 + pulsewidth_ms # offset cathodic phase 
                t2 = t1 + ipi_ms # onset anodic phase
                t3 = t2 + pulsewidth_ms # offset anodic phase
                current_changepoints_by_ch[ch_idx].append([t0, -ampl_ua*UA2PA])
                current_changepoints_by_ch[ch_idx].append([t1, 0])
                current_changepoints_by_ch[ch_idx].append([t2, ampl_ua*UA2PA])
                current_changepoints_by_ch[ch_idx].append([t3, 0])
        self.current_changepoints_by_ch = [np.array(x) for x in current_changepoints_by_ch]
        print(self.current_changepoints_by_ch)
    
    def get_current_at_locs(self, locs):
        """
        Get the current at locations at time t
        
        Parameters
        ----------
        current_changepoints_by_ch : list [n_chs] of current changespoints.
            Each list is a (n_changepoints, 2) where the first column is time 
            and the second is the current amplitude (pA)
        locs : (n_locs, 2) ndarray of target locations
        
        Returns
        -------
        current_at_locs : (n_locs) list of Nest step_current_generator objects
        """
        # first get a (n_chs, n_locs, n_times) array of currents at locations where n_times is dependeont on n_chs
        distx_ch2loc = np.subtract.outer(self.ch_coordinates[:, 0], locs[:, 0]) # (n_chs, n_locs)
        disty_ch2loc = np.subtract.outer(self.ch_coordinates[:, 1], locs[:, 1]) # (n_chs, n_locs)
        distance_ch2loc = np.sqrt(distx_ch2loc**2+disty_ch2loc**2) # (n_chs, n_locs)
        print(distance_ch2loc.shape)
        currents_ch2loc = []
        for ch_idx in range(self.n_chs):
            ch_current_vals = self.current_changepoints_by_ch[ch_idx][:, 1] # (n_changepoints)
            distances_to_locs = distance_ch2loc[ch_idx, :] # (n_locs)
            currents_to_locs = self.current_disperse_func(ch_current_vals[:, None], distances_to_locs[None, :]).T # (n_locs, n_changepoints)
            currents_ch2loc.append(currents_to_locs)
        # currents_ch2loc is [n_chs] list of (n_locs, n_changepoints) arrays
        # TODO consider using the full timestamps instead of just the change points for more regular data shape

        # then iterate over the locations and sum up the currents from all channels
        current_generators = []
        for i_loc, loc in enumerate(locs):
            current_changepoints_all = []
            for ch_idx in range(self.n_chs):
                current_times_ch = self.current_changepoints_by_ch[ch_idx][:, 0] # (n_changepoints)
                current_vals_ch = currents_ch2loc[ch_idx][i_loc, :] # (n_changepoints)
                assert len(current_times_ch) == len(current_vals_ch)
                current_changepoints_all.extend(list(zip(current_times_ch, current_vals_ch)))
            # current_changepoints_all is a [n_changpoints] list of (time, current) tuples
            current_times_reduced = []
            current_vals_reduced = [] 
            for time, currents in sort_and_groupby(current_changepoints_all, key=lambda x: round(x[0]/nest.resolution)*nest.resolution):
                current_times_reduced.append(time)
                current_vals_reduced.append(sum([current for _, current in currents]))
            # TODO eliminate the diff-0 "change points"
            # print(current_times_reduced)
            gen = nest.Create(
                "step_current_generator", 
                params=dict(
                    label="current_delivery_at_loc_%d"%(i_loc),
                    amplitude_times=current_times_reduced,
                    amplitude_values=current_vals_reduced
                )
            )
            current_generators.append(gen)
        return current_generators
