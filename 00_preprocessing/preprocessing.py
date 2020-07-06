import numpy as np
import pandas as pd
import os
#from itertools import filterfalse
from itertools import chain
from tqdm import tqdm
import json
import papermill as pm

### Grids dir
GRIDS_DIR = "/home/rio/ssh_grids"
# ssh1 grids 
SSH1_GRIDS_DIR = os.path.join(GRIDS_DIR,"ssh1")
SSH1_PERIODIC_100_6561_GRID_DIR = os.path.join(SSH1_GRIDS_DIR,"periodic_100_6561")
SSH1_PERIODIC_140_6561_GRID_DIR = os.path.join(SSH1_GRIDS_DIR,"periodic_140_6561")
SSH1_PERIODIC_180_6561_GRID_DIR = os.path.join(SSH1_GRIDS_DIR,"periodic_180_6561")
SSH1_PERIODIC_220_6561_GRID_DIR = os.path.join(SSH1_GRIDS_DIR,"periodic_220_6561")
# ssh2 grids 
SSH2_GRIDS_DIR = os.path.join(GRIDS_DIR,"ssh2")
SSH2_PERIODIC_100_6561_GRID_DIR = os.path.join(SSH2_GRIDS_DIR,"periodic_100_6561")
SSH2_PERIODIC_140_6561_GRID_DIR = os.path.join(SSH2_GRIDS_DIR,"periodic_140_6561")
SSH2_PERIODIC_180_6561_GRID_DIR = os.path.join(SSH2_GRIDS_DIR,"periodic_180_6561")
SSH2_PERIODIC_220_6561_GRID_DIR = os.path.join(SSH2_GRIDS_DIR,"periodic_220_6561")

### CSVS dir
CSVS_DIR = "/home/rio/ssh_csvs" 
SSH1_CSVS_DIR = os.path.join(CSVS_DIR,"ssh1")
SSH2_CSVS_DIR = os.path.join(CSVS_DIR,"ssh2")

### Generating directories
generate_dirs = [CSVS_DIR, SSH1_CSVS_DIR, SSH2_CSVS_DIR]
for d in generate_dirs:
    if not os.path.isdir(d):
        os.mkdir(d)   

### CSV names
# ssh1 
SSH1_PERIODIC_100_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_100_6561.csv")
SSH1_PERIODIC_140_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_140_6561.csv")
SSH1_PERIODIC_180_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_180_6561.csv")
SSH1_PERIODIC_220_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_220_6561.csv")
# ssh2 
SSH2_PERIODIC_100_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_100_6561.csv")
SSH2_PERIODIC_140_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_140_6561.csv")
SSH2_PERIODIC_180_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_180_6561.csv")
SSH2_PERIODIC_220_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_220_6561.csv")

### Output files
# ssh1
SSH1_PERIODIC_100_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh1_periodic_100_6561.ipynb"
SSH1_PERIODIC_140_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh1_periodic_140_6561.ipynb"
SSH1_PERIODIC_180_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh1_periodic_180_6561.ipynb"
SSH1_PERIODIC_220_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh1_periodic_220_6561.ipynb"
# ssh2
SSH2_PERIODIC_100_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh2_periodic_100_6561.ipynb"
SSH2_PERIODIC_140_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh2_periodic_140_6561.ipynb"
SSH2_PERIODIC_180_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh2_periodic_180_6561.ipynb"
SSH2_PERIODIC_220_6561_OUTPUT_FILE = "zzz_preprocessing_output_ssh2_periodic_220_6561.ipynb"

### Template notebook
TEMPLATE_NOTEBOOK = "0_preprocessing_template.ipynb"

### Kernel name
KERNEL_NAME = "ml_top_phases"

def load_hamiltonians(grid_folder):
    """"
    Loads hamiltonians stored in .dat files
    
    inputs
    grid_folder: a string. Name of folder in ./grid containing the grid of hamiltonians.
    n_samples: an int. Number of samples to draw. If None, all data files are sampled.

    return
    list_of_hamiltonians: a list of strings. Each string is the name of a file with data from a hamiltonian.
    raw_data: a numpy array.
    
    """
    current_dir = os.getcwd()
    grid_path = os.path.join(current_dir, "grids", grid_folder)
    os.chdir(grid_path)
    raw_data = []
    list_of_hamiltonians = np.array(os.listdir(os.getcwd()))
    raw_data = []
    for ham in tqdm(list_of_hamiltonians, desc = "loading hamiltonians"):
        with open(ham) as f:
            d = f.readlines()
            for line in d:
                raw_data.append(line.strip("\n").split("\t"))
    raw_data = np.array(raw_data)
    os.chdir(current_dir)
    return list_of_hamiltonians, raw_data

def row_major_sort(dataframe):
    """
    Sorts a dataframe with respect to t values, row-major-wise.

    input
    dataframe: a properly formatted pandas dataframe.
    """
    t1_column = dataframe.columns.get_loc("t1")
    winding_column = dataframe.columns.get_loc("winding")
    t_array = np.flip(dataframe.iloc[:,t1_column:winding_column].values, axis=1)
    n_ts = t_array.shape[1]
    for column in range(n_ts):
        sorted_args = np.argsort(t_array[:,column], kind = "mergesort")
        dataframe = dataframe.iloc[sorted_args,:]
        t_array = t_array[sorted_args]
    return dataframe
    
def make_train_test(dataframe, allowed_windings = None, epsilon = 0.01):
    """
    Creates train and test sets using winding intervals as criteria.

    input
    dataframe: a properly formatted pandas dataframe.
    allowed_windings: a list of ints with allowed winding values.
    epsilon: a float. Tolerance for numerical windings. Any data with winding out of epsilon tolerance will be made test.  
    """
    ###setting type_of columns
    winding_values = dataframe.winding.values
    rounded_windings = np.rint(winding_values)
    rounded_windings[np.isinf(rounded_windings)] = 999    #changing infinite values in rounded_windings to a large integer
    boolean_train = np.logical_and(np.abs(winding_values-rounded_windings) < epsilon, np.in1d(rounded_windings, allowed_windings)) 
    dataframe.loc[boolean_train,"type_of"] = "train"
    dataframe.loc[np.logical_not(boolean_train),"type_of"] = "test"
    is_999 = dataframe.phase == 999
    dataframe.loc[is_999,"type_of"] = "test" 
              
def make_dataframe(raw_data, list_of_hamiltonians, allowed_windings = None, epsilon = 0.01, csv_dir = None, csv_name = None):
    """
    Generates a dataframe from .dat hamiltonians
    
    input
    raw_data: a numpy array containining raw data from the grid of hamiltonians.
    list_of_hamiltonians: a list of strings. Each string is the name of a Hamiltonian.
    allowed_windings: a list of ints with allowed winding values.
    epsilon: a float. Tolerance for numerical windings. Any data with winding out of epsilon tolerance will be made test.  
    csv_name: a string. Name of csv file to be saved in ./csv.
    to_csv = a bool. Whether to write dataframe to csv file or not. 
        
    return
    dataframe: a pandas dataframe containing the data from all .dat files in the grid_path directory
    """
    non_feature_columns = 4
    n_features = raw_data.shape[1] - non_feature_columns
    n_hamiltonians = len(list_of_hamiltonians)
    n_ts = len(list_of_hamiltonians[0].replace("H_","").replace(".dat","").split("_")[:-1])
    feature_names = ["t" + str(i+1) for i in range(n_ts)] + ["winding"] + ["phase"] + ["feat" + str(i) for i in range(n_features)] 
    ### rolling dataframe
    raw_data = np.roll(raw_data,shift = 4, axis=1)
    dataframe = pd.DataFrame(data = raw_data, columns = feature_names)
    ### processing winding column
    winding = np.char.replace(dataframe.winding.values.astype(str), " ", "")
    winding = winding.astype(np.float64)
    winding[np.isnan(winding)] = np.inf
    dataframe.winding = winding
    ### processing phase column   
    dataframe.loc[:,"phase"] = dataframe.phase.astype(np.float64)
    phase_values = dataframe.phase.values
    allowed_phases = np.in1d(phase_values, allowed_windings)
    dataframe.loc[np.logical_not(allowed_phases), "phase"] = 999  #setting disallowed phase values to a large integer
    dataframe.loc[:,"phase"] = dataframe.phase.astype(np.int32)
    ### adding pred_phase column
    pred_phase_index = dataframe.columns.get_loc("phase") + 1
    dataframe.insert(pred_phase_index, "pred_phase", [666]*len(dataframe))  # filling pred_phase column with large integer
    ### adding hamiltonian path column
    path_column = np.repeat(list_of_hamiltonians, n_features)
    dataframe.insert(0, "path", path_column)
    ### adding type_of column
    dataframe.insert(dataframe.columns.get_loc("feat0"), "type_of", ["null"]*len(dataframe))
    make_train_test(dataframe, allowed_windings, epsilon)
    ### processing t columns 
    t1_column = dataframe.columns.get_loc("t1")
    winding_column = dataframe.columns.get_loc("winding")
    t_columns = dataframe.iloc[:,t1_column:winding_column].values.astype(np.float64)
    dataframe.iloc[:,t1_column:winding_column] = t_columns
    ### c-row sorting with respect to t values
    dataframe = row_major_sort(dataframe)
    ### adding hamiltonian id column
    id_column = np.repeat(np.arange(n_hamiltonians),n_features) 
    dataframe.insert(0, "id", id_column)
    ### reseting index
    dataframe.reset_index(drop=True, inplace=True)
    ### writing to csv
    if csv_dir is not None: 
        path_to_save = os.path.join(csv_dir,csv_name)
        dataframe.to_csv(path_or_buf=path_to_save)
    return dataframe

### Running preprocessing 
