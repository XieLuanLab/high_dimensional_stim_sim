# from collections import dict

import nest
import numpy as np

class Network:
    def __init__(self, n_neurons, neuron_locations, connectivity_matrix):
        self.n_neurons = n_neurons
        self.connectivity_matrix = connectivity_matrix
        self.neuron_locations = neuron_locations
        self.create_population()
        self.connect_neurons()
        self.spike_recorders = None # []
    
    def create_population(self):
        # TODO tune neurons
        # self.neurons : nest.NodeCollection = nest.Create("aeif_cond_exp", self.n_neurons)
        self.neurons : nest.NodeCollection = nest.Create("iaf_psc_alpha", self.n_neurons)
        self.neuron_ids = nest.GetStatus(self.neurons, "global_id")
        self.neuron_id2idx_dict = dict(zip(self.neuron_ids, range(self.n_neurons)))

    def connect_neurons(self):
        # TODO tune synapses
        # pre_inds, post_inds = np.nonzero(self.connectivity_matrix)
        # # n_synapses = len(pre_inds)
        # # pre_inds and post_inds should both be (n_synapses,)
        # weights = self.connectivity_matrix[pre_inds, post_inds]
        # # pre_array = np.ones(n_synapses, dtype=int)*self.neurons[pre_inds].get("global_id")
        # self.synapses = nest.Connect(
        #     self.neurons[pre_inds], self.neurons[post_inds],
        #     conn_spec="one_to_one",
        #     syn_spec={"weight": weights},
        #     return_synapsecollection=True
        # )
        noise = nest.Create("noise_generator", params={"mean": 500.0, "std": 1000.0})
        nest.Connect(noise, self.neurons, conn_spec="all_to_all")
        for pre_idx in range(self.n_neurons):
            post_inds = np.nonzero(self.connectivity_matrix[pre_idx, :])[0]
            weights = self.connectivity_matrix[pre_idx, post_inds]
            # print(weights)
            nest.Connect(
                self.neurons[pre_idx], self.neurons[post_inds],
                conn_spec="all_to_all",
                syn_spec={"weight": weights[:, None]}
            )


    def simulate_current_input(self, input_currents, time_ms, save_path=None):
        """
        Parameters
        ----------
        input_currents : (n_neurons, ) list of nest step_current_generator's
        time_ms : stimulation time
        """
        # connect the step current generators to neurons 1-on-1
        for input_current, neuron in zip(input_currents, self.neurons):
            nest.Connect(input_current, neuron)
        # add scope
        self.record_to = "memory"
        sd_dict = {
            'record_to': self.record_to,
            'label': "spk_rec"
        }
        self.spike_recorders = nest.Create(
            "spike_recorder", n=self.n_neurons,
            params=sd_dict
        )
        self.voltage_recorders = nest.Create("multimeter", n=self.n_neurons, params={"record_from":["V_m"]})
        nest.Connect(self.neurons, self.spike_recorders, conn_spec="one_to_one")
        nest.Connect(self.voltage_recorders, self.neurons, conn_spec="one_to_one")
        # simulateion
        nest.Simulate(time_ms)
    
    def get_spiketrains(self):
        assert self.spike_recorders is not None
        assert self.record_to == "memory"
        spk_trains = [[] for _ in range(self.n_neurons)]
        result_dict_tuple = self.spike_recorders.get("events")
        # result_senders = result_dict["senders"]
        # result_times = result_dict["times"]
        # assert np.all(np.diff(result_times)>=0)
        # for sender, time in zip(result_senders, result_times):
        for i_neuron, result_dict in enumerate(result_dict_tuple):
            # print(result_dict)
            times = result_dict["times"]
            if len(times) > 0:
                sender = result_dict["senders"][0]
                assert np.all(np.diff(times)>0)
                assert np.all(result_dict["senders"]==sender)
                spk_trains[i_neuron] = times
            else:
                spk_trains[i_neuron] = np.array([])
        return [np.array(x) for x in spk_trains]

    def get_voltages(self):
        assert self.voltage_recorders is not None
        assert self.record_to == "memory"
        volt_traces = [[] for _ in range(self.n_neurons)]
        result_dict_tuple = self.voltage_recorders.get("events")
        # result_senders = result_dict["senders"]
        # result_times = result_dict["times"]
        # assert np.all(np.diff(result_times)>=0)
        # for sender, time in zip(result_senders, result_times):
        for i_neuron, result_dict in enumerate(result_dict_tuple):
            # print(result_dict)
            times = result_dict["times"]
            volts = result_dict["V_m"]
            if len(times) > 0:
                sender = result_dict["senders"][0]
                assert np.all(np.diff(times)>0)
                assert np.all(result_dict["senders"]==sender)
                volt_traces[i_neuron] = (times, volts)
            else:
                raise ValueError("No voltage trace recorded")
        return volt_traces



    def simulate_spiketrain_input(self, input_spikes, time):
        """
        input_spikes : (n_neurons) list of input spike train objects; 
            where None means that neuron does not receive outside synaptic input
        """
        pass