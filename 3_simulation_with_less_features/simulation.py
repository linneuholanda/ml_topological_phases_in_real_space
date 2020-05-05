import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import graphviz
from itertools import filterfalse
from importlib import reload
from itertools import chain
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from tqdm import tqdm
from joblib import dump
import csv
import papermill as pm

### Simulation directories
SIMULATIONS_DIR = "/home/rio/ssh_simulations"
# ssh1 simulations
SSH1_SIMULATIONS_DIR = os.path.join(SIMULATIONS_DIR,"ssh1")
SSH1_PERIODIC_LESS_100_6561_SIMULATION_DIR = os.path.join(SSH1_SIMULATIONS_DIR,"periodic_less_100_6561")
SSH1_PERIODIC_LESS_140_6561_SIMULATION_DIR = os.path.join(SSH1_SIMULATIONS_DIR,"periodic_less_140_6561")
SSH1_PERIODIC_LESS_180_6561_SIMULATION_DIR = os.path.join(SSH1_SIMULATIONS_DIR,"periodic_less_180_6561")
SSH1_PERIODIC_LESS_220_6561_SIMULATION_DIR = os.path.join(SSH1_SIMULATIONS_DIR,"periodic_less_220_6561")
# ssh2 simulations
SSH2_SIMULATIONS_DIR = os.path.join(SIMULATIONS_DIR,"ssh2")
SSH2_PERIODIC_LESS_100_6561_SIMULATION_DIR = os.path.join(SSH2_SIMULATIONS_DIR,"periodic_less_100_6561")
SSH2_PERIODIC_LESS_140_6561_SIMULATION_DIR = os.path.join(SSH2_SIMULATIONS_DIR,"periodic_less_140_6561")
SSH2_PERIODIC_LESS_180_6561_SIMULATION_DIR = os.path.join(SSH2_SIMULATIONS_DIR,"periodic_less_180_6561")
SSH2_PERIODIC_LESS_220_6561_SIMULATION_DIR = os.path.join(SSH2_SIMULATIONS_DIR,"periodic_less_220_6561")

### Generating simulation directories
generate_dirs = [SIMULATIONS_DIR,
                 SSH1_SIMULATIONS_DIR,
                 SSH1_PERIODIC_LESS_100_6561_SIMULATION_DIR,
                 SSH1_PERIODIC_LESS_140_6561_SIMULATION_DIR,
                 SSH1_PERIODIC_LESS_180_6561_SIMULATION_DIR,
                 SSH1_PERIODIC_LESS_220_6561_SIMULATION_DIR,
                 SSH2_SIMULATIONS_DIR,
                 SSH2_PERIODIC_LESS_100_6561_SIMULATION_DIR,
                 SSH2_PERIODIC_LESS_140_6561_SIMULATION_DIR,
                 SSH2_PERIODIC_LESS_180_6561_SIMULATION_DIR,
                 SSH2_PERIODIC_LESS_220_6561_SIMULATION_DIR,
                ]
for d in generate_dirs:
    if not os.path.isdir(d):
        os.mkdir(d) 

### CSVS dirs
CSVS_DIR = "/home/rio/ssh_csvs" 
# ssh1 paths
SSH1_CSVS_DIR = os.path.join(CSVS_DIR,"ssh1")
SSH1_PERIODIC_100_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_100_6561.csv")
SSH1_PERIODIC_140_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_140_6561.csv")
SSH1_PERIODIC_180_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_180_6561.csv")
SSH1_PERIODIC_220_6561_CSV = os.path.join(SSH1_CSVS_DIR,"periodic_220_6561.csv")
# ssh2 paths
SSH2_CSVS_DIR = os.path.join(CSVS_DIR,"ssh2")
SSH2_PERIODIC_100_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_100_6561.csv")
SSH2_PERIODIC_140_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_140_6561.csv")
SSH2_PERIODIC_180_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_180_6561.csv")
SSH2_PERIODIC_220_6561_CSV = os.path.join(SSH2_CSVS_DIR,"periodic_220_6561.csv")

### Output files
# ssh1
SSH1_PERIODIC_LESS_100_6561_OUTPUT_FILE = "zzz_simulation_output_ssh1_periodic_less_100_6561.ipynb"
SSH1_PERIODIC_LESS_140_6561_OUTPUT_FILE = "zzz_simulation_output_ssh1_periodic_less_140_6561.ipynb"
SSH1_PERIODIC_LESS_180_6561_OUTPUT_FILE = "zzz_simulation_output_ssh1_periodic_less_180_6561.ipynb"
SSH1_PERIODIC_LESS_220_6561_OUTPUT_FILE = "zzz_simulation_output_ssh1_periodic_less_220_6561.ipynb"
# ssh2
SSH2_PERIODIC_LESS_100_6561_OUTPUT_FILE = "zzz_simulation_output_ssh2_periodic_less_100_6561.ipynb"
SSH2_PERIODIC_LESS_140_6561_OUTPUT_FILE = "zzz_simulation_output_ssh2_periodic_less_140_6561.ipynb"
SSH2_PERIODIC_LESS_180_6561_OUTPUT_FILE = "zzz_simulation_output_ssh2_periodic_less_180_6561.ipynb"
SSH2_PERIODIC_LESS_220_6561_OUTPUT_FILE = "zzz_simulation_output_ssh2_periodic_less_220_6561.ipynb"

### Template notebook
TEMPLATE_NOTEBOOK = "0_simulation_template.ipynb"

### Kernel name
KERNEL_NAME = "ml_top_phases"

class Simulation(object):
    """
    Implements a machine learning simulation 
    """    
    _models_dict = {"DecisionTreeClassifier": DecisionTreeClassifier, "RandomForestClassifier": RandomForestClassifier}
    def __init__(self, csv_path, model_name, model_kw, allowed_windings, simulation_dir = None, val_split = 0, features_to_use = None, shuffle_features = False, random_state = None):            
        """
        Simulation class constructor.

        input
        csv_path: a string. Path to csv file to be loaded.
        model: a scikit-learn function with fit/predict methods.
        allowed_windings: a list of ints with allowed winding values.
        simulation_dir: a string. Path to dir where simulation results will be saved.
        val_split: a float. Fraction of training data to be used in validation.
        features_to_use: a list of ints. Contains features to be used in training/predicting. If None, uses all features.
        shuffle_features: a bool. Whether to shuffle features or not.
        random_state: an int. Seed for random number generation.
        """
        ##### storing simulation parameters for reference #####
        self.parameters = locals()
        del self.parameters["self"]
        if random_state is not None:
            np.random.seed(random_state)
        self.csv_path = csv_path
        self.model = None
        self.model_name = model_name
        self.model_kw = model_kw
        self.allowed_windings = allowed_windings
        self.val_split = val_split
        self.features_to_use = features_to_use
        self.shuffle_features = shuffle_features
        self.random_state = random_state
        self.simulation_dir = simulation_dir
        ##### Building Dataframe #####
        dtype = {"id":np.int32, "path": str, "winding": np.float64, "phase": np.int32, "pred_phase": np.int32, "type_of": str}
        self.dataframe = pd.read_csv(filepath_or_buffer = self.csv_path, index_col = 0, dtype = dtype)
        self.n_features = len(self.dataframe.columns[self.dataframe.columns.get_loc("feat0"):])
        if shuffle_features:
            self.shuffle_features_array = np.random.permutation(np.arange(self.n_features))
            self.parameters["shuffle_features_array"] = self.shuffle_features_array
        self.n_hamiltonians = len(self.dataframe)//self.n_features
        self.n_ts = len(self.dataframe.columns[self.dataframe.columns.get_loc("t1"):self.dataframe.columns.get_loc("winding")])
        self.train_ids = list(np.unique(self.dataframe.id[self.dataframe.type_of == "train"].values))
        self.test_ids = list(np.unique(self.dataframe.id[self.dataframe.type_of == "test"].values))
        self.val_ids = []
        ##### Simulation results #####
        self.eigenvector_columns = ["id", "phase", "pred_phase", "type_of"]
        self.eigenvector_summary = None
        self.hamiltonian_summary = None
        self.accuracy = {"eigenvector_train": None, "eigenvector_val": None, "eigenvector_test": None, "hamiltonian_train": None, "hamiltonian_val": None, "hamiltonian_test": None}
        self.eigenvector_summary_list = []
        self.hamiltonian_summary_list = []
        self.accuracy_list = {"eigenvector_train": [], "eigenvector_val": [], "eigenvector_test": [], "hamiltonian_train": [], "hamiltonian_val": [], "hamiltonian_test": []}

    def make_val(self):
        """
        Creates validation set from training data
        """
        #print("inside make_val!!!")
        n_train_val = len(self.train_ids)+len(self.val_ids)
        n_val = int(n_train_val*self.val_split)
        n_train = n_train_val - n_val
        shuffle = np.random.permutation(n_train_val)
        train_val_ids = np.array(self.train_ids + self.val_ids)
        new_train_ids = list(train_val_ids[shuffle][:n_train])
        new_val_ids = list(train_val_ids[shuffle][n_train:])
        ### updating type_of column
        is_train = np.in1d(self.dataframe.id.values, new_train_ids) 
        self.dataframe.loc[is_train,"type_of"] = "train"
        is_val = np.in1d(self.dataframe.id.values, new_val_ids) 
        self.dataframe.loc[is_val,"type_of"] = "val"
        ### updating train and val ids
        self.train_ids = new_train_ids
        self.val_ids = new_val_ids
        #print("leaving make_val!!!")

    def set_features_to_use(self,features):
        """
        Updates features used in simulations

        input
        features: a list of features to be used.
        """
        self.features_to_use = features

    def fit(self, fit_params = None, shuffle_rows=True):
        """
        Fits model to eigenvectors with fit_params parameters

        input
        fit_params: a dict of fitting parameters.
        shuffle_rows: a bool. Whether to shuffle rows before fitting.
        """
        train_rows = self.dataframe.type_of == "train"
        feat_columns = self.dataframe.columns[self.dataframe.columns.get_loc("feat0"):]
        X, y = self.dataframe.loc[train_rows,feat_columns].values, self.dataframe[train_rows].phase.values
        ### shuffling features
        if self.shuffle_features:
            X = X[:,self.shuffle_features_array]
        ### selecting features
        if self.features_to_use is not None:
            if self.shuffle_features:
                column_ix = np.argwhere(np.in1d(self.shuffle_features_array, self.features_to_use)).reshape(-1,)
            else:
                column_ix = self.features_to_use
            X = X[:,column_ix]
            #c = np.argwhere(np.in1d(b, np.array([6,3,5,4,9]))).reshape(-1,)
        ### shuffling training data
        if shuffle_rows:
            shuffle = np.random.permutation(len(X)) 
            X = X[shuffle]
            y = y[shuffle]
        if fit_params is None:
            fit_params = {"X": X, "y": y}    
        else:
            fit_params["X"] = X; fit_params["y"] = y
        self.model.fit(**fit_params)
        #print("X: ", X)
        #print("shape of X: ", X.shape)

    def predict(self, predict_params = None):
        """
        Uses fitted model to predict on eigenvectorswith pred_params
    
        input
        dataframe: a pandas dataframe with properly named columns
        predict_params: a dict of prediction parameters
        """ 
        print("THIS IS  predict_params: ", predict_params)
        print("THIS IS  the type of predict_params: ", type(predict_params))
        feat_columns = self.dataframe.columns[self.dataframe.columns.get_loc("feat0"):]
        X = self.dataframe.loc[:,feat_columns].values
        ### shuffling features
        if self.shuffle_features:
            X = X[:,self.shuffle_features_array]
        ### selecting features
        if self.features_to_use is not None:
            if self.shuffle_features:
                column_ix = np.argwhere(np.in1d(self.shuffle_features_array, self.features_to_use)).reshape(-1,)
            else:
                column_ix = self.features_to_use
            X = X[:,column_ix]
        if predict_params is None:
            predict_params = {"X": X}
        else:
            predict_params["X"] = X
        y_pred = self.model.predict(**predict_params)
        self.dataframe.loc[:,"pred_phase"] = y_pred

    def predict_hamiltonians(self):
        """
        Predicts Hamiltonians' phases through majority voting of eigenvectors (has to be fitted and predicted first!)
        """
        ### majority voting (breaks ties randomly)
        vote_counts = pd.crosstab(self.dataframe.id, self.dataframe.pred_phase, normalize = "index")
        votes_array = vote_counts.values
        max_values = np.max(votes_array, axis = 1).reshape((-1,1))
        boolean_max = np.equal(votes_array, max_values)
        elected_list = []
        for i in range(len(boolean_max)):
            args = np.argwhere(boolean_max[i,:]).reshape((-1,))
            elected_arg = np.random.choice(args)
            elected_list.append(self.allowed_windings[elected_arg])
        vote_counts["pred_phase"] = elected_list
        ### adding phase column
        true_phases = pd.crosstab(self.dataframe.id, self.dataframe.phase).idxmax(axis=1)
        phase_index = vote_counts.columns.get_loc("pred_phase")
        vote_counts.insert(phase_index, "phase", true_phases)
        ### adding t columns
        indices = [i*self.n_features for i in range(self.n_hamiltonians)]
        columns = self.dataframe.columns[self.dataframe.columns.get_loc("t1"):self.dataframe.columns.get_loc("winding")]
        t_df = self.dataframe.loc[indices, columns]
        t_df.reset_index(drop=True, inplace = True)
        ### adding type_of column
        type_of = self.dataframe.loc[indices, "type_of"].values
        t_df["type_of"] = type_of
        ### concatenating dataframes
        summary = pd.concat((t_df,vote_counts),axis=1)
        summary.index.name = "id"
        return summary
    
    def compute_accuracy(self):
        """
        Computes accuracies of a fitted and predicted model
        """
        #eigenvector_summary = self.eigenvector_summary[-1]
        #hamiltonian_summary = self.hamiltonian_summary[-1]
        ### eigenvector_train accuracy 
        boolean_mask = self.eigenvector_summary["type_of"]=="train"
        y_true = self.eigenvector_summary.phase[boolean_mask].values
        y_pred = self.eigenvector_summary.pred_phase[boolean_mask].values
        self.accuracy["eigenvector_train"] = accuracy_score(y_true,y_pred)
        ### eigenvector_val accuracy 
        #print("Before boolean mask!!")
        boolean_mask = self.eigenvector_summary["type_of"]=="val"
        #print("boolean_mask in compute_accuracy!!!")
        if np.sum(boolean_mask) > 0:
            y_true = self.eigenvector_summary.phase[boolean_mask].values
            y_pred = self.eigenvector_summary.pred_phase[boolean_mask].values
            self.accuracy["eigenvector_val"] = accuracy_score(y_true,y_pred)
        else: 
            self.accuracy["eigenvector_val"] = None
        ### eigenvector_test accuracy 
        boolean_mask = np.logical_and(self.eigenvector_summary["type_of"]=="test", np.in1d(self.eigenvector_summary.phase, self.allowed_windings))
        y_true = self.eigenvector_summary.phase[boolean_mask].values
        y_pred = self.eigenvector_summary.pred_phase[boolean_mask].values
        self.accuracy["eigenvector_test"] = accuracy_score(y_true,y_pred)
        ### hamiltonian_train accuracy 
        boolean_mask = self.hamiltonian_summary["type_of"]=="train"
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.accuracy["hamiltonian_train"] = accuracy_score(y_true,y_pred)
        ### hamiltonian_val accuracy 
        #print("boolean_mask for hamiltonian_val")
        boolean_mask = self.hamiltonian_summary["type_of"]=="val"
        if np.sum(boolean_mask) > 0:
            y_true = self.hamiltonian_summary.phase[boolean_mask].values
            y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
            self.accuracy["hamiltonian_val"] = accuracy_score(y_true,y_pred)
        else: 
            self.accuracy["hamiltonian_val"] = None
        #print("boolean_mask just after hamiltonian_val")
        ### hamiltonian_test accuracy 
        boolean_mask = np.logical_and(self.hamiltonian_summary["type_of"]=="test", np.in1d(self.hamiltonian_summary.phase, self.allowed_windings))
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.accuracy["hamiltonian_test"] = accuracy_score(y_true,y_pred)
                 
    def run_simulation(self, n_experiments=1, start_n=0, fit_params=None, shuffle_rows = True, pred_params=None, random_features = False, store_in_lists=False, save_eigenvector=False, save_hamiltonian=True, save_accuracy=True, save_models=False):
        """
        Fits a machine learning algorithm to training data and writes result of simulation to disk
        
        input
        n_experiments: an int. Number of experiments to perform.
        start_n: an int. Id number of first simulation.
        fit_params: a dict of fitting parameters
        shuffle_rows: a bool. Whether to shuffle rows before fitting.
        pred_params: a dict of prediction parameters
        random_features: int. Number of random features to use.
        store_in_lists: a bool. Whether to store results from simulations in lists
        save_eigenvector: a bool. Whether to save eigenvector summaries in disk.
        save_hamiltonian: a bool. Whether to save hamiltonian summaries in disk.
        save_accuracy: a bool. Whether to save accuracy summaries in disk.
        save_models: a bool. Whether to save models in disk.
        """
        if save_eigenvector or save_hamiltonian or save_accuracy or save_models:
            eigenvector_path = os.path.join(self.simulation_dir, "eigenvector")
            hamiltonian_path = os.path.join(self.simulation_dir, "hamiltonian")
            accuracy_path = os.path.join(self.simulation_dir, "accuracy")
            model_path = os.path.join(self.simulation_dir, "model")
            if not os.path.isdir(self.simulation_dir):
                os.mkdir(self.simulation_dir)
            if (save_eigenvector) and not os.path.isdir(eigenvector_path):
                os.mkdir(eigenvector_path)
            if (save_hamiltonian) and not os.path.isdir(hamiltonian_path):
                os.mkdir(hamiltonian_path) 
            if (save_accuracy) and not os.path.isdir(accuracy_path):
                os.mkdir(accuracy_path)
            if (save_models) and not os.path.isdir(model_path):
                os.mkdir(model_path)
            with open(os.path.join(self.simulation_dir, "parameters.csv"), 'w') as f:  
                    w = csv.writer(f)
                    w.writerows(self.parameters.items())
        for exp in tqdm(range(start_n,start_n+n_experiments), desc = "running experiments"):
            ### creating model
            self.model = Simulation._models_dict[self.model_name](**self.model_kw)
            ### picking features randomly 
            if random_features:
                self.features_to_use = np.random.randint(0,self.n_features,size=random_features)
                print("random_features: ", self.features_to_use)
            ### making validation sets
            self.make_val()
            ### fitting and predicting
            self.fit(fit_params, shuffle_rows)
            self.predict(pred_params)
            ### generating simulation summaries
            self.eigenvector_summary = self.dataframe[self.eigenvector_columns]  
            self.hamiltonian_summary = self.predict_hamiltonians()          
            self.compute_accuracy()
            ### storing results
            if store_in_lists:
                self.eigenvector_summary_list.append(self.eigenvector_summary)
                self.hamiltonian_summary_list.append(self.hamiltonian_summary)
                for key, value in self.accuracy.items():
                    self.accuracy_list[key].append(value) 
            ### dumping to disk
            filename = str(exp)
            if save_eigenvector:
                self.eigenvector_summary.to_csv(path_or_buf=os.path.join(eigenvector_path, filename + ".csv") )
            if save_hamiltonian:
                self.hamiltonian_summary.to_csv(path_or_buf=os.path.join(hamiltonian_path, filename + ".csv") )
            if save_accuracy:
                with open(os.path.join(accuracy_path, filename + ".csv"), 'w') as f:  
                    w = csv.writer(f)
                    w.writerows(self.accuracy.items())
            if save_models:
                dump(self.model, os.path.join(model_path, filename + ".joblib") )
               
    def visualize_scatter_2d(self, fig_params={}, val_params={}, test_params={}, train_params={}, legend_params={}, xlabel_params={}, ylabel_params={}, title_params={}, savefig_params = {}):
        """
        Creates a scatter plot visualization of train/val/test data. Use after run_simulation().

        input
        fig_params: a dict with parameters for plt.fig.
        val_params: a dict with parameters for plt.scatter with val data.
        test_params: a dict with parameters for plt.scatter with test data.
        train_params: a dict with parameters for plt.scatter with train data.
        legend_params: a dict with parameters for plt.legend.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        ### setting train data
        train_filter = self.hamiltonian_summary["type_of"] == "train" 
        t1_train = self.hamiltonian_summary.t1[train_filter].values
        t2_train = self.hamiltonian_summary.t2[train_filter].values
        train_params["x"] = t1_train
        train_params["y"] = t2_train
        ### setting val data
        val_filter = self.hamiltonian_summary["type_of"] == "val" 
        t1_val = self.hamiltonian_summary.t1[val_filter].values
        t2_val = self.hamiltonian_summary.t2[val_filter].values
        val_params["x"] = t1_val
        val_params["y"] = t2_val
        ### setting test data
        test_filter = self.hamiltonian_summary["type_of"] == "test" 
        t1_test = self.hamiltonian_summary.t1[test_filter].values
        t2_test = self.hamiltonian_summary.t2[test_filter].values
        test_params["x"] = t1_test
        test_params["y"] = t2_test
        ### plotting
        figure = plt.figure(**fig_params)
        plt.scatter(**val_params)
        plt.scatter(**test_params)
        plt.scatter(**train_params)
        plt.legend(**legend_params)
        plt.xlim(np.min(self.hamiltonian_summary.t1.values), np.max(self.hamiltonian_summary.t1.values))
        plt.ylim(np.min(self.hamiltonian_summary.t2.values), np.max(self.hamiltonian_summary.t2.values))
        if len(xlabel_params) > 0:
            plt.xlabel(**xlabel_params)
        if len(ylabel_params) > 0:
            plt.ylabel(**ylabel_params)
        if len(title_params) > 0:
            plt.title(**title_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
        #plt.show()


#################### visualization ####################

def visualize_tree(tree_clf, tree_name, feature_names, class_names, destination = "trees"):
    """
    Creates a visualization of a decision tree using graphviz

    inputs
    tree_clf: a tree classifer
    tree_name: a string
    feature_names: a list of strings
    class_names: a list of strings
    destination: a string (path to folder where files will be saved). If non existant, will be created
    """
    if not os.path.isdir(destination):
        path = os.path.join(os.getcwd(), destination)
        print("Creating directory " + path + "\n")
        os.makedirs(path)
    os.chdir(path)
    dot_data = export_graphviz(tree_clf, out_file = None, feature_names = feature_names, filled = True, rounded = True, special_characters = True)
    graph = graphviz.Source(dot_data)
    graph.render(tree_name)
    os.chdir("..")
    return graph



    
