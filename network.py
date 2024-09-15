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
        self.simulation_time_ms = 0
    
    def create_population(self):     
        # https://nest-simulator.readthedocs.io/en/stable/auto_examples/gif_population.html
        neuron_params = {
            "C_m": 83.1,
            "g_L": 3.7,
            "E_L": -67.0,
            "Delta_V": 1.4,
            "V_T_star": -39.6,
            "t_ref": 4.0,
            "V_reset": -36.7, 
            "lambda_0": 1, # Baseline firing intensity
            
            # Stronger spike-triggered current (more inhibition after each spike)
            "q_stc": [50, -10],
            "tau_stc": [100, 200],
            
            # Stronger spike-frequency adaptation (higher threshold after each spike)
            "q_sfa": [20, 2],
            "tau_sfa": [50, 1000],
        }
        
        self.neurons : nest.NodeCollection = nest.Create("gif_psc_exp", self.n_neurons, neuron_params)
        self.neuron_ids = nest.GetStatus(self.neurons, "global_id")
        self.neuron_id2idx_dict = dict(zip(self.neuron_ids, range(self.n_neurons)))

    def connect_neurons(self):
        # simulate natural firing 
        w_noise = 1000.0  # synaptic weights from Gamma to population neurons
        gamma_generator = nest.Create("gamma_sup_generator", params={
            "rate": 10, # Mean rate of input
            "gamma_shape": 2,  # Regularity of input (higher value = more regular)
        })
        nest.Connect(gamma_generator, self.neurons, conn_spec="all_to_all", syn_spec={"weight": w_noise})
        
        # simulate noise in membrane potential
        white_noise = nest.Create("noise_generator", params={"mean": 100.0, "std": 1000.0})  
        nest.Connect(white_noise, self.neurons, syn_spec={"weight": 0.7})
        
        # connect neurons 
        for pre_idx in range(self.n_neurons):
            post_inds = np.nonzero(self.connectivity_matrix[pre_idx, :])[0]
            weights = self.connectivity_matrix[pre_idx, post_inds]
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
        self.voltage_recorders = nest.Create("multimeter", n=self.n_neurons, params={"record_from": ["V_m"]})
        nest.Connect(self.neurons, self.spike_recorders, conn_spec="one_to_one")
        nest.Connect(self.voltage_recorders, self.neurons, conn_spec="one_to_one")
        # simulateion
        self.simulation_time_ms = time_ms
        nest.Simulate(time_ms)
        
    def simulate_baseline(self, time_ms, save_path=None):
        self.record_to = "memory"
        sd_dict = {
            'record_to': self.record_to,
            'label': "spk_rec"
        }
        self.spike_recorders = nest.Create(
            "spike_recorder", n=self.n_neurons,
            params=sd_dict
        )
        self.voltage_recorders = nest.Create("multimeter", n=self.n_neurons, params={"record_from": ["V_m"]})
        nest.Connect(self.neurons, self.spike_recorders, conn_spec="one_to_one")
        nest.Connect(self.voltage_recorders, self.neurons, conn_spec="one_to_one")
        self.simulation_time_ms = time_ms
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

