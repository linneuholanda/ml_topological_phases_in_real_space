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
#import json
import csv
   
class Simulation(object):
    """
    Implements a machine learning simulation 
    """    
       
    def __init__(self, csv_name, model, allowed_windings, val_split = 0, simulation_name = None, random_state = None):            
        """
        Simulation class constructor.

        input
        csv_name: a string. Name of csv file to be loaded from ./csv.
        model: a scikit-learn function with fit/predict methods.
        allowed_windings: a list of ints with allowed winding values.
        val_split: a float. Fraction of training data to be used in validation.
        simulation_name: a string. Name to use when saving files.
        random_state: an int. Seed for random number generation.
        """
        ##### storing simulation parameters for reference #####
        self.parameters = locals()
        del self.parameters["self"]
        if random_state is not None:
            np.random.seed(random_state)
        self.csv_name = csv_name
        self.model = model
        self.allowed_windings = allowed_windings
        self.val_split = val_split
        self.simulation_name = simulation_name
        self.random_state = random_state
        ##### Building Dataframe #####
        dtype = {"id":np.int32, "path": str, "winding": np.float64, "phase": np.int32, "pred_phase": np.int32, "type_of": str}
        self.dataframe = pd.read_csv(filepath_or_buffer = os.path.join("./csv", csv_name), index_col = 0, dtype = dtype)
        self.n_features = len(self.dataframe.columns[self.dataframe.columns.get_loc("feat1"):])
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

        ##### Loading Hamiltonians #####
        #self.hamiltonians = np.array(os.listdir(grid_path))
        #self.raw_data = load_hamiltonians(grid_path)
        ##### Building id-t dictionaries #####
        #self.t_to_ids, self.ids_to_t = build_dictionaries(self.hamiltonians)
        ##### Building Dataframe #####
        #self.dataframe = make_dataframe(self.raw_data, self.hamiltonians)
        
        ### other attributes
        #self.majority_vote = None
        
        #self.majority_vote_train = None
        #self.majority_vote_test = None
        #self.alg_params = None
        #self.grid_params = None
        
        #self.accuracy_scores = None
        #self.eigen_train_accuracy = None
        #self.eigen_val_accuracy = None
        #self.eigen_test_accuracy = None
        #self.train_accuracy = None
        #self.val_accuracy = None
        #self.test_accuracy = None
        #self.grid_name = None
        #self.dict_t_elected = None
        #self.dict_t_true = None

    def make_val(self):
        """
        Creates validation set from training data
        """
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

    def fit(self, fit_params = None):
        """
        Fits model to eigenvectors with fit_params parameters

        input
        fit_params: a dict of fitting parameters
        """
        train_rows = self.dataframe.type_of == "train"
        feat_columns = self.dataframe.columns[self.dataframe.columns.get_loc("feat1"):]
        X, y = self.dataframe.loc[train_rows,feat_columns].values, self.dataframe[train_rows].phase.values
        ### shuffling training data
        shuffle = np.random.permutation(len(X)) 
        X = X[shuffle]
        y = y[shuffle]
        if fit_params is None:
            fit_params = {"X": X, "y": y}    
        else:
            fit_params["X"] = X; fit_params["y"] = y
        self.model.fit(**fit_params)

    def predict(self, predict_params = None):
        """
        Uses fitted model to predict on eigenvectorswith pred_params
    
        input
        dataframe: a pandas dataframe with properly named columns
        predict_params: a dict of prediction parameters
        """ 
        feat_columns = self.dataframe.columns[self.dataframe.columns.get_loc("feat1"):]
        X = self.dataframe.loc[:,feat_columns].values
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
        boolean_mask = self.eigenvector_summary["type_of"]=="val"
        y_true = self.eigenvector_summary.phase[boolean_mask].values
        y_pred = self.eigenvector_summary.pred_phase[boolean_mask].values
        self.accuracy["eigenvector_val"] = accuracy_score(y_true,y_pred)
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
        boolean_mask = self.hamiltonian_summary["type_of"]=="val"
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.accuracy["hamiltonian_val"] = accuracy_score(y_true,y_pred)
        ### hamiltonian_test accuracy 
        boolean_mask = np.logical_and(self.hamiltonian_summary["type_of"]=="test", np.in1d(self.hamiltonian_summary.phase, self.allowed_windings))
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.accuracy["hamiltonian_test"] = accuracy_score(y_true,y_pred)
                 
    def run_simulation(self, n_experiments=1, start_n=0, fit_params=None, pred_params=None, store_in_lists=False, save_eigenvector=False, save_hamiltonian=True, save_accuracy=True, save_model=False):
        """
        Fits a machine learning algorithm to training data and writes result of simulation to disk
        
        input
        n_experiments: an int. Number of experiments to perform.
        start_n: an int. Id number of first simulation.
        fit_params: a dict of fitting parameters
        pred_params: a dict of prediction parameters
        store_in_lists: a bool. Whether to store results from simulations in lists
        save_eigenvector: a bool. Whether to save eigenvector summaries in disk.
        save_hamiltonian: a bool. Whether to save hamiltonian summaries in disk.
        save_accuracy: a bool. Whether to save accuracy summaries in disk.
        save_model: a bool. Whether to save models in disk.
        """
        if save_eigenvector or save_hamiltonian or save_accuracy or save_models:
            path = os.path.join("./simulation", self.simulation_name)
            eigenvector_path = os.path.join(path, "eigenvector")
            hamiltonian_path = os.path.join(path, "hamiltonian")
            accuracy_path = os.path.join(path, "accuracy")
            model_path = os.path.join(path, "model")
            if not "simulation" in os.listdir(os.getcwd()):
                os.mkdir("simulation")
            if not self.simulation_name in os.listdir("simulation"):
                os.mkdir(path)
                #results_path = os.path.join("./simulation_data", self.simulation_name)
            if (save_eigenvector) and not "eigenvector" in os.listdir(path):
                os.mkdir(eigenvector_path)
            if (save_hamiltonian) and not "hamiltonian" in os.listdir(path):
                os.mkdir(hamiltonian_path) 
            if (save_accuracy) and not "accuracy" in os.listdir(path):
                os.mkdir(accuracy_path)
            if (save_model) and not "model" in os.listdir(path):
                os.mkdir(model_path)
            with open(os.path.join(path, "parameters.csv"), 'w') as f:  
                    w = csv.writer(f)
                    w.writerows(self.parameters.items())
        for exp in tqdm(range(start_n,start_n+n_experiments), desc = "running experiments"):
            ### making validation sets
            self.make_val()
            ### fitting and predicting
            self.fit(fit_params)
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
            if save_model:
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
                        
                    
               
            
        #storing simulation parameters for reference
        
        #self.summary = {**locals(), **self.summary}
        #del self.summary["self"]
        #print("locals: ", locals())
        #print("\n\n\n")
        #print("self.summary: ", self.summary)
        #print("\n\n\n")
        #self.model = Simulation.algorithms[self.algorithm](alg_params, grid_params)
        

        #self.fit(fit_params)
        #self.predict(pred_params)
        

        #print("self  majority tran: \n", self.majority_vote_train)
        #print("self  majority test: \n", self.majority_vote_test)
        #self.majority_vote = pd.concat((self.majority_vote_train, self.majority_vote_test), axis=0)
        ####HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.majority_vote = self.model.majority_vote(self.dataframe, normalize, tiebreaker)
        #self.eigen_train_accuracy = (self.train_dataframe.phase == self.train_dataframe.pred_phase).mean()
        #self.eigen_test_accuracy = (self.test_dataframe.phase == self.test_dataframe.pred_phase).mean()
        

        ### Computing accuracies
        #self.eigen_train_accuracy = (self.dataframe[ self.dataframe["type_of"] == "train" ].phase == self.dataframe[ self.dataframe["type_of"] == "train" ].pred_phase).mean()
        #self.eigen_val_accuracy = (self.dataframe[ self.dataframe["type_of"] == "val" ].phase == self.dataframe[ self.dataframe["type_of"] == "val" ].pred_phase).mean()
        #self.eigen_test_accuracy = (self.dataframe[ self.dataframe["type_of"] == "test" ].phase == self.dataframe[ self.dataframe["type_of"] == "test" ].pred_phase).mean()
        #self.eigen_test_accuracy = (self.dataframe[ self.dataframe["type_of"] == "test" ].phase == self.dataframe[ self.dataframe["type_of"] == "test" ].pred_phase).mean()
        #self.eigen_test_accuracy = None
       # self.train_accuracy = (self.majority_vote[self.majority_vote["type_of"] == "train"].elected == self.majority_vote[self.majority_vote["type_of"] == "train"].true).mean()
        #self.val_accuracy = (self.majority_vote[self.majority_vote["type_of"] == "val"].elected == self.majority_vote[self.majority_vote["type_of"] == "val"].true).mean()
        #self.test_accuracy = None
        #self.test_accuracy = (self.majority_vote[self.majority_vote["type_of"] == "test"].elected == self.majority_vote[self.majority_vote["type_of"] == "test"].true).mean()
        #self.train_accuracy = (self.majority_vote_train.elected == self.majority_vote_train.true).mean()
        #self.test_accuracy = (self.majority_vote_test.elected == self.majority_vote_test.true).mean()
        #self.accuracy_scores = {"eigen_train": self.eigen_train_accuracy, "eigen_val": self.eigen_val_accuracy,"eigen_test": self.eigen_test_accuracy, "train": self.train_accuracy, "val": self.val_accuracy, "test": self.test_accuracy}
        #self.train_accuracy = (self.majority_vote_train.elected == self.majority_vote_train.true).mean()
        #self.test_accuracy = (self.majority_vote_test.elected == self.majority_vote_test.true).mean()
        #self.scores = {"eigen_train_accuracy": self.eigen_train_accuracy, "eigen_test_accuracy": self.eigen_test_accuracy,
        #               "train_accuracy": self.train_accuracy, "test_accuracy": self.test_accuracy}
        #print("self.summary: ", self.summary)
        #### fill in majority_vote_test

    def grid_predict(self):
        """
        Generates numpy arrays suitable for plotting in 2d parameter space 

        input
       
        return 
        grid_arrays: an array containing the grid arrays xx, yy, zz
        """
        #t_values_train = np.array(self.majority_vote_train.t_values.values.tolist()) 
        #t_values_test = np.array(self.majority_vote_test.t_values.values.tolist()) 
        #t_values = np.vstack((t_values_train, t_values_test))
        
        t_values = np.array(self.majority_vote.t_values.values.tolist())
        #print("shape of t_values: ", t_values.shape)
        #t_values = self.majority_vote.t_values.values
        #print("type of t_values: ", type(t_values))
        #t_values = np.array(t_values.tolist())
        #print("here!!!\n")
        #print("t_values: ", t_values)
        #print("type of t_values: ", type(t_values))
        #print("shape of t_values: ", t_values.shape)
        t1, t2 = np.unique(t_values[:,0]), np.unique(t_values [:,1])
        #print("t1: ", t1)
        #print("t2: ", t2)
        xx, yy = np.meshgrid(t1,t2)
        #print("xx: ", xx)
        #print("yy: ", yy)
        zz = []

       #######################################################################

        for point in np.c_[xx.ravel(), yy.ravel()]:
            #print("point: ", point)
            #print("tuple(point): ", tuple(point))
            print("tuple(point): ", tuple(point))
            print("point: ", point)
            #print("t_values: ", self.majority_vote["t_values"])
            #print("point: ", point)
            ix = self.majority_vote.index[self.majority_vote["t_values"] == tuple(point)]
            print("ix: ", ix)
            print("checking equality: " )
            if len(ix) != 0:
                zz.append(self.majority_vote.loc[ix, "elected"].values[0])
            else: 
                zz.append(np.inf)

        ######################################################################
        #for point in np.c_[xx.ravel(), yy.ravel()]:
        #    id_ = self.t_values_id[tuple(point)]
        #    print("point: ", point)
        #    print("id_: ", id_)
        #    print("correct spot in majority_vote: ", self.majority_vote.loc[id_, "t_values"])
            
        zz = np.array(zz).reshape(xx.shape)
        return xx, yy, zz
        #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        #    Z = Z.reshape(xx.shape)
        #    cs = pl.contourf(xx, yy, Z)
        #print("self.t_values_id: \n", )
        #print(self.t_val




        
        #for t in np.c_[xx.ravel(), yy.ravel()]:
        #    print("t: ", t)
        #    print("selected: ")
            #print(self.majority_vote[self.majority_vote.t_values == tuple(t)])
            #zz.append( self.majority_vote.elected[np.equal(self.majority_vote.t_values, t)])
        #    id_ = self.t_values_id[tuple(t)]
        #    print("self.majority_vote.id == id_ : ", elf.majority_vote[self.majority_vote.id == id_])
        #    zz.append( self.majority_vote[self.majority_vote.id == id_].elected.values[0] )
        #zz = np.array(zz)
        #zz = self.model.model.predict(np.c_[xx.ravel(), yy.ravel()])
        #zz = zz.reshape(xx.shape)
        #cs = plt.countourf(xx,yy,zz)
        #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        #    Z = Z.reshape(xx.shape)
        #    cs = pl.contourf(xx, yy, Z)
        
        
        #pass
    
    def save(self, simulation_name = None):
        """
        Saves simulationt disk

        input
        simulation_name: a string. Path to folder where files will be saved
        """
        #self.summary = {**locals(), **self.summary}
        #del self.summary["self"]
        if simulation_name is not None:
            self.summary["simulation_name"] = simulation_name
        current_dir = os.getcwd()
      
        grid_name = self.grid_path.split("/")[-1]
        path_to_results = "./results"
        
        if not grid_name in os.listdir(path_to_results):
            os.mkdir(os.path.join(path_to_results,grid_name))
            os.chdir(os.path.join(path_to_results,grid_name))
            results_dirs = ["train_dataframe", "test_dataframe", "train_votes", "test_votes", "scores", "summary"]
            for d in results_dirs:
                os.mkdir(d)
        else:
            os.chdir(os.path.join(path_to_results,grid_name))
        
        self.train_dataframe.to_csv(os.path.join("./train_dataframe", simulation_name))
        self.test_dataframe.to_csv(os.path.join("./test_dataframe", simulation_name))
        self.majority_vote_train.to_csv(os.path.join("./train_votes", simulation_name))
        self.majority_vote_test.to_csv(os.path.join("./test_votes", simulation_name))
        with open(os.path.join("./scores", simulation_name), "w") as f:#####################################3
            json.dump(self.scores, f)
        with open(os.path.join("./summary", simulation_name), "w") as f:#####################################3
            json.dump(self.summary, f)   
        os.chdir(current_dir)

    #def predict
    
    def make_plot2d(self, grid, eigen_probabilities = False, scatter_train = False):
        """
        Makes a 2d plot of experiment

        input
        grid: a tuple of numpy arrays (xarray, yarray). These will be used to generate a meshgrid for the values of the t's
        eigen_probabilities: a bool. Whether to use eigenvectors votes as probabilities or not
        scatter_train: a bool. Whether to plot training data as scatter plot or not
        """  
        xarray, yarray = grid[0], grid[1]
        xx, yy = np.meshgrid(xarray, yarray)
        X = np.c_[xx, yy]
        Z = self.model.model.predict(X)
        plt.contourf(xx,)
        
        
        
        
        
        

        
        
       
        

class MonteCarlo(object):
    """
    Performs Monte Carlo simulations to determine accuracy of algorithms
    """
    alg_dict = {"DecisionTrees": 1}
    def __init__(self, n_exp, algorithm, data_dir = "../grids/periodic/ssh_t1_t2"):
        """
        MonteCarlo class constructor

        input
        n_exp: an int. Number of experiments to be performed
        algorithm: a string mapping to a ml algorithm (possibly from scikit-learn). Contains methods like "fit" and "predict"
        data_dir: a string. Path to folder with hamiltonians
        """
        self.n_exp = n_exp
        self.algorithm = algorithm
        self.data_dir = data_dir

    def experiment(self, train_data, test_data):
        """
        ...    
        """
        pass 
    




#################### metrics ####################

def majority_vote(dataframe, y_vector, unique_phases = np.arange(4), normalize = "index"):
    """
    Performs majority vote on the eigenvectors  
    
    input
    dataframe: a pandas dataframe
    y_vector: a numpy array with phases
    unique_phases: a numpy array with the unique phases 
    normalize: a boolean variable flagging whether vote counts should be normalized or not
    return
    elected: a dictionary of elected phases
    votes: a dictionary of percentage votes
    """
  
    votes = pd.crosstab(dataframe.hamiltonian_number, y_vector.astype(int), normalize = normalize)
    votes.columns.name = "phase"
    for p in unique_phases:
        if not p in votes.columns:
            votes.insert(int(p), p, np.zeros(len(votes)).astype(int)  )
    elected = votes.idxmax(axis = 1) 
       
    return votes, elected
    


def accuracy_ham(elected_true, elected_pred):
    """
    Computes accuracy scores for full hamiltonians

    input
    elected_true: a numpy array with true phases
    elected_pred: a numpy array predicted phases

    return
    accuracy: a float
    """
    return np.mean(elected_true == elected_pred)
    
    
    





        

    
   

   
   

  






#################### visualization ####################

def plot_hamiltonians2d(phase_dict, figsize = (10,10)):
    """
    Makes a scatter plot of hamiltonians
    
    input
    phase_dict: a dictionary mapping tuples of t to their phases
    figsize: figsize to plot function
    """
    return None



def plot_feature_importances(tree_clf, n_features = "all", return_arrays = False):
    """
    Plots feature importances for tree based classifiers

    input
    tree_clf: a tree based classifier
    n_features: number of features to show in plot. If "all", all are showed
    return_arrays: a boolean variable. Whether return importance arrays or not

    return 
    sorted_indices: array of features in decreasing order of importance
    sorted_importances: array of importances in decreasing order
    cumulative_importances: array of cumulative importances computed from sorted_importances
    """
    importances = tree_clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    cumulative_importances = np.cumsum(sorted_importances)
    sorted_features = np.array([str(ix) for ix in sorted_indices ])
    if n_features == "all":
        n_features = len(sorted_indices)
    fig, ax = plt.subplots(2,1, figsize = (30,50))
    for ix, feat in enumerate(sorted_features[:n_features]):
        ax[0].bar(feat, sorted_importances[ix])
        ax[0].set_title("Feature importances", size = 24)
        ax[0].set_xlabel("Feature", fontsize = 24)
        ax[0].set_ylabel("Importance", fontsize = 24)
        ax[1].bar(feat, cumulative_importances[ix])
        ax[1].set_title("Cumulative feature importances", size = 24)
        ax[1].set_xlabel("Feature", fontsize = 24)
        ax[1].set_ylabel("Cumulative importance ", size = 24)
   
    plt.show()
    if return_arrays:
        return sorted_indices, sorted_importances, cumulative_importances
    else:
        return None

        





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



    
