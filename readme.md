# high_dimensional_stim_sim

This code repo provides simulation of __high-dimensional stimulation__'s effects on local neurons in a cortical column model.
The simulation is used to demonstrate our point of view that, by applying high-dimensional stimulation, we can indeed approach the high dimensionality of evoked neural activities similar to those during spontaneous/natural behaviors. 

## TO RUN
### Simulation scripts
- run_deterministic_stim_corcol_sim.py  -- run simulations with deterministic stim on random neuronal circuits.
- run_random_stim_corcol_sim.py         -- run simulations with random stim on circuits mimicking a layered cortical column (TODO: perform dim reduc on readout; needs a little bit additional time because the output spike train format is different from `run_random_stim_sim.py`)

### Analysis scripts
- plot_psths.py                 --  plot peri-stimulus time histograms of simulated stimulation pulses.
- analyze_different_configs.py  -- Analyze and compare stimulation patterns with different dimensionalities

### Dependency
- NEST simulation environment
- Numpy/Scipy/Matplotlib
