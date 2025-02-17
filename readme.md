# High-Dimensional Stimulation Simulation
 
This repository provides code for simulating high-dimensional stimulation and its effects on local neurons in a cortical column model comprising 3858 neurons. 
The simulation is based on the Potjans and Diesmann (2014) cortical column model and implemented using NEST 3.8. 
Our results demonstrate that intracortical multielectrode stimulation can evoke neural activity patterns with increasing dimensionality, approaching the dimensionality of spontaneous/natural activity.

## Zenodo Dataset for Reproducing Figures
The dataset used to generate figures in our publication is available at:  
📥 [10.5281/zenodo.14880297](https://zenodo.org/records/14880298)


## Installation and Setup
Dependencies:
- NEST simulator 
- NumPy, SciPy, Matplotlib, SciKit-Learn

## Installation  (Windows via WSL2)
1. Install WSL2 (Ubuntu recommended)
2. Install Miniconda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
3. Create a conda environment and install dependencies:
```
conda create --name stim_sim -c conda-forge nest-simulator numpy scipy matplotlib scikit-learn
conda activate stim_sim
```
4. (Optional) Install Spyder. 
`conda install spyder`
5. Clone the repository.
```
git clone https://github.com/oaaij-gnahz/high_dimensional_stim_sim.git
cd high_dimensional_stim_sim
```

## Recreating Figures from Publication
1. Download the dataset from Zenodo: 
📥 [10.5281/zenodo.14880297](https://zenodo.org/records/14880298)
2. Create a folder inside the cloned repository:
```
mkdir data
mkdir figures
```
3. Copy/paste the downloaded dataset into the newly created `data` folder.
4. Run plot_figures_for_sim_condition.py.

## Running a New Simulation from Scratch
1. Modify parameters in configuration files inside corcol_params/.
2. Adjust simulation settings in run_random_stim_corcol_sim.py.
3. Run run_random_stim_corcol_sim.py.

