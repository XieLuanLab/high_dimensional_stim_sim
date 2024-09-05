# stim_sim

Very rudimentary simulation of __high-dimensional stimulation__'s effects on local neurons.

# General idea
- Generate a neuronal population by specifying coordinates and connectivity matrix
- Define probe layout, electrode locations, and how the current amplitude delivered from electrodes decay in space (e.g. use a Gaussian function)
- Each neuron receives input from the pulse trains (after decay) and from other neurons within the population. (External input not implmented yet)
- Simulate random/spontaneous input to population and obtain the resulting population response profile (M, N, T) matrices where M is the number of experiments, N is the number of neurons and T number of time bins. This can be viewed as a dataset of M sample each containing NxT features.
- Obtain dataset of same format for when stimulation is delivered.
- Compared the spontaneous and stim'd datasets.
- What we desire to see: By giving specific stimulation patterns, we can drive the population activity to some specific region in some embedded space or mimic the output induced by some specific external "spontaneous" input.

# Script structure
- run_sim.py    -- main script to run
- electrodes.py -- define class Electrodes
- network.py    -- define neuronal populations
- utils.py      -- other utility functions


## Dependency
- NEST simulation environment
- Numpy/Scipy/Matplotlib

# NOTES/TODOs
- TODO generate random current input as background noise
- TODO determine what "spontaneous" input looks like
- NOTE Currently not considering the morphology of neurons (may be doable by using the compartmental model) nor the electrode area|size.
- TODO Need to tune the neuron parameters such as synaptic weights, conductance, etc.; also need to tune the decay of stimulation current as a function of distance to neurons. (The scales of these values could stray away from actual experimental results, but as long as the result holds up, that should not be a critical concern?)
