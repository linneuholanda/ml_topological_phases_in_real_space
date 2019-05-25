# Machine learning topological phases in real space

This repository contains Python scripts and notebooks for the paper ["Machine learning topological phases in real space"]
(https://arxiv.org/abs/1901.01963). 

## Dependencies

The most straightforward way to run this code is by setting up a Python [virtual environment](https://docs.python.org/3/library/venv.html) in a Linux machine. Set up your Python environment following the instructions below.

1. Clone repository to local directory and cd into it.
```bash
git clone https://github.com/linneuholanda/ml_topological_phases_in_real_space.git /path/to/local/directory
cd /path/to/local/directory
```
2. Create and activate a new environment with Python 3.6.
    ```bash
    python3.6 -m venv ml_topological
    source ml_topological/bin/activate
    ``` 
3. Move to the repository root folder and install the requirements.
   ```bash
   pip install -r requirements.txt
   ```
 ## Repository files
 
 The repository contains the following script/notebook pairs:
 
 i) preprocessing.py/generate_csv.ipynb 
 ii) simulation.py/run_simulation.ipynb
 ii) prediction_grid.py/generate_grids.ipynb
 
 At each step the notebook uses the corresponding script to process .mat files into a single .csv file (step i), run machine learning experiments (step ii) and plot visualizations (step iii).
 
 ## Data
 
As explained in the paper, the data used in each experiment (SSH with nearest-neighbour hoppings and SSH with first and second nearest-neighbours hoppings) consist of real space eigenvectors of 6,561 Hamiltonians. We provide links to download the data below. 

a) [Data for SSH systems with nearest-neighbour hoppings]
b) [Data for SSH systems with first and second neighbours hoppings S]

Extract the files in `/path/to/local/directory` and run the notebooks i)-iii). Once you are finished, deactivate the Python environment with

``` bash
deactivate
```
