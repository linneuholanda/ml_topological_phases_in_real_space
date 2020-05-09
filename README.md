# Machine learning topological phases in real space

This repository contains Python scripts and notebooks for the paper ["Machine learning topological phases in real space"]
(https://arxiv.org/abs/1901.01963). 

## Dependencies

The most straightforward way to run this code is by setting up a Python [virtual environment](https://docs.python.org/3/library/venv.html) in a Linux machine. Set up your Python environment following the instructions below.

1. Create and activate a new environment with Python 3.6.
    ```bash
    python3.6 -m venv ~/ml_topological
    source ~/ml_topological/bin/activate
    ``` 
2. Clone repository to local directory and cd into it.
```bash
git clone https://github.com/linneuholanda/ml_topological_phases_in_real_space.git /path/to/local/directory
cd /path/to/local/directory
```
3. Install the requirements.
   ```bash
   pip install -r requirements.txt
   ```
 ## Repository files
 
 The repository contains the following ordered directories:
 
 0_preprocessing 
 
 1_simulation

 2_results

 3_simulation_with_less_features

 4_results_with_less_features

 5_paper

 6_arxiv

 7_prb

 8_prb_submission
 
 Run the numbered notebooks in directories 0-4 in order to generate the results in the paper. Directory 5 contains a template for the paper. Directory 6 contains the Arxiv submission. Directory 7 contains a Revtex template for the *Physical Review B*. Directory 8 contains the PRB submission. 
 
 ## Data
 
As explained in the paper, the data used in each experiment (SSH with nearest-neighbour hoppings and SSH with first and second nearest-neighbours hoppings) consist of real space eigenvectors of 6,561 Hamiltonians. We provide links to download the data below. 

a) [Data for SSH systems with nearest-neighbour hoppings](https://www.dropbox.com/s/h5pzbibt1z3zda6/nearest_neighbour_SSH_6561.rar?dl=0)

b) [Data for SSH systems with first and second neighbours hoppings](https://www.dropbox.com/s/zmkfacu53p583na/first_and_second_neighbours_SSH_6561.rar?dl=0)

Extract the files in the proper directory and run the notebooks. Once you are finished, deactivate the Python environment with

``` bash
deactivate
```
