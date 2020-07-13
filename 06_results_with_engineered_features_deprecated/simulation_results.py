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
from joblib import load
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
#import json
import csv

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

### Paper directory
FIGURES_DIR = "/home/rio/ml_topological_phases_in_real_space/paper"
# ssh1 figures
SSH1_FIGURES_DIR = os.path.join(FIGURES_DIR,"ssh1")
SSH1_PERIODIC_LESS_100_6561_FIGURES_DIR = os.path.join(SSH1_FIGURES_DIR,"periodic_less_100_6561")
SSH1_PERIODIC_LESS_140_6561_FIGURES_DIR = os.path.join(SSH1_FIGURES_DIR,"periodic_less_140_6561")
SSH1_PERIODIC_LESS_180_6561_FIGURES_DIR = os.path.join(SSH1_FIGURES_DIR,"periodic_less_180_6561")
SSH1_PERIODIC_LESS_220_6561_FIGURES_DIR = os.path.join(SSH1_FIGURES_DIR,"periodic_less_220_6561")
# ssh2 figures
SSH2_FIGURES_DIR = os.path.join(FIGURES_DIR,"ssh2")
SSH2_PERIODIC_LESS_100_6561_FIGURES_DIR = os.path.join(SSH2_FIGURES_DIR,"periodic_less_100_6561")
SSH2_PERIODIC_LESS_140_6561_FIGURES_DIR = os.path.join(SSH2_FIGURES_DIR,"periodic_less_140_6561")
SSH2_PERIODIC_LESS_180_6561_FIGURES_DIR = os.path.join(SSH2_FIGURES_DIR,"periodic_less_180_6561")
SSH2_PERIODIC_LESS_220_6561_FIGURES_DIR = os.path.join(SSH2_FIGURES_DIR,"periodic_less_220_6561")

### Generating figure directories
generate_dirs = [FIGURES_DIR,
                 SSH1_FIGURES_DIR,
                 SSH1_PERIODIC_LESS_100_6561_FIGURES_DIR,
                 SSH1_PERIODIC_LESS_140_6561_FIGURES_DIR,
                 SSH1_PERIODIC_LESS_180_6561_FIGURES_DIR,
                 SSH1_PERIODIC_LESS_220_6561_FIGURES_DIR,
                 SSH2_FIGURES_DIR,
                 SSH2_PERIODIC_LESS_100_6561_FIGURES_DIR,
                 SSH2_PERIODIC_LESS_140_6561_FIGURES_DIR,
                 SSH2_PERIODIC_LESS_180_6561_FIGURES_DIR,
                 SSH2_PERIODIC_LESS_220_6561_FIGURES_DIR,
                ]
for d in generate_dirs:
    if not os.path.isdir(d):
        os.mkdir(d)

### Output notebooks
# ssh1
SSH1_PERIODIC_LESS_100_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh1_periodic_less_100_6561.ipynb"
SSH1_PERIODIC_LESS_140_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh1_periodic_less_140_6561.ipynb"
SSH1_PERIODIC_LESS_180_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh1_periodic_less_180_6561.ipynb"
SSH1_PERIODIC_LESS_220_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh1_periodic_less_220_6561.ipynb"
# ssh2
SSH2_PERIODIC_LESS_100_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh2_periodic_less_100_6561.ipynb"
SSH2_PERIODIC_LESS_140_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh2_periodic_less_140_6561.ipynb"
SSH2_PERIODIC_LESS_180_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh2_periodic_less_180_6561.ipynb"
SSH2_PERIODIC_LESS_220_6561_OUTPUT_NOTEBOOK = "zzz_results_output_ssh2_periodic_less_220_6561.ipynb"

############### Defining ListedColorMaps
ssh1_colormap = ListedColormap(["hotpink", "lightskyblue"], name="ssh1")
ssh2_colormap = ListedColormap(["hotpink", "orange", "lightgreen","lightskyblue"], name="ssh2") 
plt.register_cmap(cmap=ssh1_colormap)
plt.register_cmap(cmap=ssh2_colormap)

############### Defining some color maps
###Reds_alpha
ncolors = 256
color_array = plt.get_cmap('Reds')(range(ncolors))
# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='Reds_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)


###Oranges_alpha
ncolors = 256
color_array = plt.get_cmap('Oranges')(range(ncolors))
# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='Oranges_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)


### Greens_alpha
ncolors = 256
color_array = plt.get_cmap('Greens')(range(ncolors))
# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='Greens_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)

### Blues_alpha
ncolors = 256
color_array = plt.get_cmap('Blues')(range(ncolors))
# change alpha values
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='Blues_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)
##########

class Experiment(object):
    """
    A class to perform data analysis on machine learning experiments.
    """

    def __init__(self, simulation_dir, experiment_name):
        """
        ExperimentAnalysis class constructor. 

        input
        simulation_dir: a string. Path to parent simulation directory.
        experiment_name: a string. One of the experiments in simulation_dir.
        """
        #self.working_dir = os.getcwd()
        self.simulation_dir = simulation_dir
        self.experiment_name = experiment_name
        self.eigenvector_summary = pd.read_csv(os.path.join(self.simulation_dir, "eigenvector", experiment_name + ".csv"), index_col = 0)
        self.hamiltonian_summary = pd.read_csv(os.path.join(self.simulation_dir, "hamiltonian", experiment_name + ".csv"), index_col = 0)
        self.accuracy = {}
        with open(os.path.join(self.simulation_dir, "accuracy", experiment_name + ".csv")) as f:
            for row in f:
                key, value = row.split(",")
                self.accuracy[key] = float(value)
                #print("row: ", row)
                #print("row[0]: ", row[0])
                #print("row[1]: ", row[1])
            #self.accuracy = {row[0]: float(row[1]) for row in f}
        self.model = load(os.path.join(self.simulation_dir, "model", experiment_name + ".joblib"))
        ### grid arrays
        self.allowed_windings = [int(w) for w in self.hamiltonian_summary.columns[self.hamiltonian_summary.columns.get_loc("type_of")+1: self.hamiltonian_summary.columns.get_loc("phase")] ] ### int(w)
        self.t1 = np.unique(self.hamiltonian_summary.t1.values)
        self.t2 = np.unique(self.hamiltonian_summary.t2.values)
        self.xx, self.yy = np.meshgrid(self.t2, self.t1)
        self.grid_shape = self.xx.shape
        self.prediction_grid = self.hamiltonian_summary.pred_phase.values.reshape(self.grid_shape)
        self.winding_grid = {w: self.hamiltonian_summary[str(w)].values.reshape(self.grid_shape) for w in self.allowed_windings}
        ### some useful statistics
        self.n_hamiltonians = len(self.hamiltonian_summary)
        self.n_train_hamiltonians = len(self.hamiltonian_summary[self.hamiltonian_summary.type_of == "train"])
        self.n_val_hamiltonians = len(self.hamiltonian_summary[self.hamiltonian_summary.type_of == "val"])
        self.n_test_hamiltonians = len(self.hamiltonian_summary[self.hamiltonian_summary.type_of == "test"])

    def print_train_val_test(self):
        """
        Prints statistics about train/val/test splits.
        """
        print("% train: ", self.n_train_hamiltonians/self.n_hamiltonians)
        print("% val: ",  self.n_val_hamiltonians/self.n_hamiltonians)
        print("% test: ", self.n_test_hamiltonians/self.n_hamiltonians)
        print("% train + val + test: ", (self.n_train_hamiltonians+self.n_val_hamiltonians+self.n_test_hamiltonians)/self.n_hamiltonians)
        print("\n")
        print("number of train hamiltonians: ", self.n_train_hamiltonians)
        print("number of val hamiltonians: ", self.n_val_hamiltonians)
        print("number of test hamiltonians: ", self.n_test_hamiltonians)
        print("total number of hamiltonians: ", self.n_hamiltonians)

    def print_phases(self, exclude_999 = True):
        """
        Prints statistics about labels in sets.
        """
        train_val_filter = ~(self.hamiltonian_summary["type_of"]=="test")
        print("Summary of train + val data\n")
        print("% train/val phases: ")
        print(self.hamiltonian_summary.phase[train_val_filter].value_counts(normalize=True))
        print("\n")
        print("total train/val phases: ")
        print(self.hamiltonian_summary.phase[train_val_filter].value_counts(normalize=False))
        if exclude_999:
            test_filter = (self.hamiltonian_summary["type_of"]=="test") & ~(self.hamiltonian_summary["phase"] == 999)
        else:
            test_filter = self.hamiltonian_summary["type_of"]=="test"
        print("\n")
        print("Summary of test data\n")
        print("% test phases: ")
        print(self.hamiltonian_summary.phase[test_filter].value_counts(normalize=True))
        print("\n")
        print("total test phases: ")
        print(self.hamiltonian_summary.phase[test_filter].value_counts(normalize=False))
        print("\n")
        print ("Summary of train + val + test data\n")
        if exclude_999:
            all_filter = ~(self.hamiltonian_summary.phase == 999)
        else:
            all_filter = [True]*len(self.hamiltonian_summary.phase)
        print("% phases: ")
        print(self.hamiltonian_summary.phase[all_filter].value_counts(normalize=True))
        print("\n")
        print("total phases: ")
        print(self.hamiltonian_summary.phase[all_filter].value_counts(normalize=False))

    def create_plot(self, fig_params, xlabel_params, ylabel_params, title_params, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}):
        """
        Creates a figure for plotting.
         
        input
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        
        return
        figure: a figure with properly set axes for plotting.ListedColormap(["hotpink", "lightskyblue"])
        """
        figure = plt.figure(**fig_params)
        #if fit_axes:    
        #    plt.xlim(np.min(self.t2), np.max(self.t2))
        #    plt.ylim(np.min(self.t1), np.max(self.t1))
        if len(xlim_params) > 0:
            plt.xlim(**xlim_params)
        if len(ylim_params) > 0:
            plt.ylim(**ylim_params)
        if len(xticks_params) > 0:
            plt.xticks(**xticks_params)
        if len(yticks_params) > 0:
            plt.yticks(**yticks_params)
        if len(xlabel_params) > 0:
            plt.xlabel(**xlabel_params)
        if len(ylabel_params) > 0:
            plt.ylabel(**ylabel_params)
        if len(title_params) > 0:
            plt.title(**title_params)
        #plt.show()
        return figure 

    def scatter_train_val_test(self, val_params={}, test_params={}, train_params={}, legend_params={}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params = {}):
        """
        Creates a scatter plot visualization of train/val/test data for a single experiment. Use after run_simulation().

        input
        val_params: a dict with parameters for plt.scatter with val data.
        test_params: a dict with parameters for plt.scatter with test data.
        train_params: a dict with parameters for plt.scatter with train data.
        legend_params: a dict with parameters for plt.legend.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        ### setting train data
        train_filter = self.hamiltonian_summary["type_of"] == "train" 
        t1_train = self.hamiltonian_summary.t1[train_filter].values
        t2_train = self.hamiltonian_summary.t2[train_filter].values
        train_params["x"] = t2_train
        train_params["y"] = t1_train
        ### setting val data
        val_filter = self.hamiltonian_summary["type_of"] == "val" 
        t1_val = self.hamiltonian_summary.t1[val_filter].values
        t2_val = self.hamiltonian_summary.t2[val_filter].values
        val_params["x"] = t2_val
        val_params["y"] = t1_val
        ### setting test data
        test_filter = self.hamiltonian_summary["type_of"] == "test" 
        t1_test = self.hamiltonian_summary.t1[test_filter].values
        t2_test = self.hamiltonian_summary.t2[test_filter].values
        test_params["x"] = t2_test
        test_params["y"] = t1_test
        ### plotting
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        plt.scatter(**val_params)
        plt.scatter(**test_params)
        plt.scatter(**train_params)
        plt.legend(**legend_params)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
            
    def scatter_winding_train(self, winding_params={}, legend_params={}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Generates a scatter plot of training data with winding labels for a single experiment.

        input
        winding_params: a dict of dicts. Each dict contains parameters for scatter plots of points from each winding.
        legend_params: a dict with parameters for plt.legend.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        for winding in self.allowed_windings:
            winding_train_filter = np.logical_and(self.hamiltonian_summary.phase.values == winding, self.hamiltonian_summary.type_of=="train")
            t1 = self.hamiltonian_summary.t1[winding_train_filter].values
            t2 = self.hamiltonian_summary.t2[winding_train_filter].values
            winding_params[winding]["x"] = t2
            winding_params[winding]["y"] = t1
            plt.scatter(**winding_params[winding])
        plt.legend(**legend_params)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
        
    def pcolormesh_prediction_grid(self, winding_params, pcolormesh_params, legend_params, fig_params, xlabel_params, ylabel_params, title_params, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Plots a prediction grid for a single experiment.

        input
        winding_params: a dict of dicts. Each dict contains parameters for scatter plots of points from each winding.
        legend_params: a dict with parameters for plt.legend.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        ### pcolormesh params
        #pcolormesh_params["X"] = self.xx
        #pcolormesh_params["Y"] = self.yy
        #pcolormesh_params["C"] = self.prediction_grid
        ### plotting
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        #print("pcolormesh_params: ", pcolormesh_params)
        #plt.pcolormesh(**pcolormesh_params)
        #plt.pcolormesh(self.xx,self.yy, self.prediction_grid, cmap= ListedColormap(["hotpink", "lightskyblue"]), alpha = 0.5)
        plt.pcolormesh(self.xx,self.yy, self.prediction_grid, **pcolormesh_params)
        ### making scatter plots
        for winding in self.allowed_windings:
            winding_train_filter = np.logical_and(self.hamiltonian_summary.pred_phase.values == winding, self.hamiltonian_summary.type_of=="train")
            t1 = self.hamiltonian_summary.t1[winding_train_filter].values
            t2 = self.hamiltonian_summary.t2[winding_train_filter].values
            winding_params[winding]["x"] = t2
            winding_params[winding]["y"] = t1
            plt.scatter(**winding_params[winding])
        plt.legend(**legend_params)
        #plt.colorbar()
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)

##############################################################These methods may be used in the Simulation Class as well
    def imshow_winding_grid(self, winding, imshow_params, colorbar_params={}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Plots a heatmap of windings using imshow.

        input
        winding: an int. One of the allowed windings.
        imshow_params: a dict with parameters.
        colorbar_params: a dict with parameters for plt.colorbar.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        if len(fig_params) > 0:
            figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        #figure = plt.figure(figsize=(10,10))
        winding_grid = self.winding_grid[winding]
        imshow_params["X"] = winding_grid
        plt.imshow(**imshow_params)
        if len(colorbar_params) > 0:
            labelsize = None
            if "labelsize" in colorbar_params:
                labelsize = colorbar_params.pop("labelsize")
            cbar = plt.colorbar(**colorbar_params)
            if labelsize is not None:
                cbar.ax.tick_params(labelsize=labelsize)
                colorbar_params["labelsize"] = labelsize
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
  
    #def merge_imshow_winding_grids(self, winding_params, colorbar_params, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
#        """
#        Produces a merged heatmap of windings with imshow.#

#        input
#        winding_params: a dict of dicts. Each dict contains parameters for the imshow heatmap of a single winding.
#        colorbar_params: a dict with parameters for plt.colorbar.
#        fig_params: a dict with parameters for plt.fig.
#        xlabel_params: a dict with parameters for plt.xlabel.
#        ylabel_params: a dict with parameters for plt.ylabel.
#        title_params: a dict with parameters for plt.title.
#        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.       
#        """    
#        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, #xticks_params, yticks_params)
#        for winding in self.allowed_windings:
#            imshow_params = winding_params[winding]
#            self.imshow_winding_grid(winding, imshow_params, colorbar_params = {})
#        #    winding_grid = self.winding_grid[winding]
#        #    imshow_params = winding_params[winding]
#        #    imshow_params["X"] = winding_grid
#        #    plt.imshow(**imshow_params)
#        if tight_params is not None:
#            plt.tight_layout(**tight_params)
#        if len(savefig_params) > 0:
#            plt.savefig(**savefig_params)  

    def merge_imshow_winding_grids(self, winding_params, colorbar_params, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Produces a merged heatmap of windings with imshow.

        input
        winding_params: a dict of dicts. Each dict contains parameters for the imshow heatmap of a single winding.
        colorbar_params: a dict with parameters for plt.colorbar.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.       
        """    
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        for winding in self.allowed_windings:
            if not winding in winding_params.keys():
                print("Skipping winding {}".format(str(winding)))
                continue
            print("Plotting winding {}".format(str(winding)))
            imshow_params = winding_params[winding]
            cb_params={}
            if winding in colorbar_params:
                cb_params = colorbar_params[winding]
            #print("cb_params: ", cb_params)

            ##############################################
            winding_grid = self.winding_grid[winding]
            imshow_params["X"] = winding_grid
            #print("unique values: ", np.unique(winding_grid))
            #print("max value: ", np.max(winding_grid))
            plt.imshow(**imshow_params)
            if len(cb_params) > 0:
                labelsize = None
                if "labelsize" in cb_params:
                    labelsize = cb_params.pop("labelsize")
                cbar = plt.colorbar(**cb_params)
                if "ticks" in cb_params:
                    cbar.set_ticks(cb_params["ticks"])
                if labelsize is not None:
                    cbar.ax.tick_params(labelsize=labelsize)
                    cb_params["labelsize"] = labelsize
            ##############################################
            #plt.tight_layout()
            #self.imshow_winding_grid(winding, imshow_params, colorbar_params = cb_params)
        #if len(colorbar_params) > 0:
        #    labelsize = None
        #    if "labelsize" in colorbar_params:
        #        labelsize = colorbar_params.pop("labelsize")
        #    cbar = plt.colorbar(**colorbar_params)
        #    if labelsize is not None:
        #        cbar.ax.tick_params(labelsize=labelsize)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)  

    def contourf_prediction_grid(self, winding_params, legend_params, fig_params, xlabel_params, ylabel_params, title_params, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Produces a contourf prediction grid of windings.

        input
        winding_params: a dict of dicts. Each dict contains parameters for the contourf plot of a single winding.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        for winding in self.allowed_windings:
            winding_grid = self.winding_grid[winding]
            contourf_params = winding_params[winding]
            plt.contourf(self.xx, self.yy, winding_grid, **contourf_params)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)

class ExperimentEnsemble(object):
    """
    A class to perform data analysis on an ensemble of machine learning experiments.
    """
    def __init__(self, simulation_dir, n_experiments = None, load_hamiltonian_summary = False):
        """
        SimulationAnalysis class constructor.

        input
        simulation_dir: a string. Name of dir with simulation results.
        n_experiments: Number of experiments to consider. If None, all experiments in simulation_dir will be considered.
        load_hamiltonian_summary: a bool. Whether to load hamiltonian_summary csv from disk.
        """
        ### Storing simulation dirs
        self.simulation_dir = simulation_dir
        self.accuracy_summary_dir = os.path.join(self.simulation_dir, "accuracy")
        self.eigenvector_summary_dir = os.path.join(self.simulation_dir, "eigenvector")
        self.hamiltonian_summary_dir = os.path.join(self.simulation_dir, "hamiltonian")
        self.model_dir = os.path.join(self.simulation_dir, "model")
        if n_experiments is not None:
            self.n_experiments = n_experiments
        else:
            self.n_experiments = len(os.listdir(self.hamiltonian_summary_dir))
        ### Setting current experiment to 0
        self.current_experiment = Experiment(simulation_dir=self.simulation_dir, experiment_name=str(0))  
        ### setting t1, t2, allowed windings, xx, yy and grid_shape
        self.allowed_windings = self.current_experiment.allowed_windings
        #self.allowed_windings_str = [str(w) for w in self.allowed_windings]
        self.t1 = np.unique(self.current_experiment.hamiltonian_summary.t1.values)
        self.t2 = np.unique(self.current_experiment.hamiltonian_summary.t2.values)
        self.xx, self.yy = np.meshgrid(self.t2,self.t1)
        self.grid_shape = self.xx.shape
        ### Storing accuracy results
        self.mean_accuracy = {}       #Stores mean accuracies through all experiments
        self.bootstrap_accuracy = {}
        ### Initializing hamiltonian_summary and winding_grid
        if load_hamiltonian_summary:
            self.hamiltonian_summary = pd.read_csv(os.path.join(self.simulation_dir, "hamiltonian_summary.csv"), index_col = 0)
            self.winding_grid = {w: self.hamiltonian_summary[str(w)].values.reshape(self.grid_shape) for w in self.allowed_windings}
        else:
            self.hamiltonian_summary = self.current_experiment.hamiltonian_summary
            self.hamiltonian_summary.loc[:,[str(w) for w in self.allowed_windings]] = 0
            self.hamiltonian_summary.loc[:,"pred_phase"] = 666
            self.winding_grid = {}
        ### feature importances
        self.feature_importance = None
        self.cumulative_feature_importance = None
          
    def reset_current_experiment(self):
        """
        Resets self.current_experiment to 0th experiment 
        """ 
        self.current_experiment = Experiment(simulation_dir=self.simulation_dir, experiment_name=str(0)) 
    
    def reset_hamiltonian_summary(self):
        """
        Resets self.hamiltonian_summary to initial state
        """ 
        self.reset_current_experiment()  
        self.hamiltonian_summary = self.current_experiment.hamiltonian_summary
        self.hamiltonian_summary.loc[:,[str(w) for w in self.allowed_windings]] = 0
        self.hamiltonian_summary.loc[:,"pred_phase"] = 666        
        self.winding_grid = {}

    def compute_hamiltonian_summary(self, save_to_disk = False):
        """
        Generates a mean Hamiltonian summary for all experiments.

        input 
        save_to_disk: a bool. Whether to save hamiltonian_summary to disk
        """    
        ### Computing winding fractions
        self.reset_hamiltonian_summary()
        #winding_fractions = self.hamiltonian_summary.loc[:,[str(w) for w in self.allowed_windings]].values
        winding_fractions = []
        for exp in tqdm(range(self.n_experiments), desc="mean hamiltonian summary"):
            current_hamiltonian_summary = pd.read_csv(os.path.join(self.simulation_dir, "hamiltonian", str(exp)+".csv"), index_col = 0)
            winding_fractions.append(current_hamiltonian_summary.loc[:,[str(w) for w in self.allowed_windings]].values)
        winding_stack = np.dstack(winding_fractions)
        votes_array = np.mean(winding_stack, axis = 2)
        #print("shape of votes_array: ", votes_array.shape)
        #print("votes_array: ", votes_array)
        ###########################        COME TO HERE!!!! it works!!!         ##########################
        self.hamiltonian_summary.loc[:,[str(w) for w in self.allowed_windings]] = votes_array   
        ### Majority vote (breaks ties randomly)
        max_values = np.max(votes_array, axis = 1).reshape((-1,1))
        boolean_max = np.equal(votes_array, max_values)
        elected_list = []
        for i in tqdm(range(len(boolean_max)), desc ="majority vote"):
            args = np.argwhere(boolean_max[i,:]).reshape((-1,))
            elected_arg = np.random.choice(args)
            elected_list.append(self.allowed_windings[elected_arg])
        self.hamiltonian_summary["pred_phase"] = elected_list
        ### Setting winding_grid
        self.winding_grid = {w: self.hamiltonian_summary[str(w)].values.reshape(self.grid_shape) for w in self.allowed_windings}
        if save_to_disk:
            self.hamiltonian_summary.to_csv(path_or_buf=os.path.join(self.simulation_dir, "hamiltonian_summary.csv"))

    def compute_mean_accuracy(self, save_to_disk = False):
        """
        Computes statistics through all experiments.
        """
        accuracies_dict = {"eigenvector_train": [], "eigenvector_val": [], "eigenvector_test": [], "hamiltonian_train": [], "hamiltonian_val": [], "hamiltonian_test": []}
        for exp in tqdm(range(self.n_experiments),desc="computing mean accuracies"):
            self.current_exp = Experiment(simulation_dir=self.simulation_dir, experiment_name=str(exp))
            #keys = ["eigenvector_train", "eigenvector_val", "eigenvector_test", "hamiltonian_train", "hamiltonian_val", "hamiltonian_test"]
            for k in accuracies_dict.keys():
                accuracies_dict[k].append(self.current_exp.accuracy[k])
        for k in accuracies_dict:
            self.mean_accuracy[k] = np.mean(accuracies_dict[k])
        if save_to_disk:
            with open(os.path.join(self.simulation_dir, "mean_accuracy.csv"), 'w') as f:  
                w = csv.writer(f)
                w.writerows(self.mean_accuracy.items())

    def compute_bootstrap_accuracy(self, save_to_disk = False):
        """
        Computes hamiltonian prediction accuracy using all averages from all experiments

        input
        save_to_disk: a bool. Whether to save bootstrap accuracy to disk or not. 
        """    
        ### hamiltonian_train accuracy 
        boolean_mask = self.hamiltonian_summary["type_of"]=="train"
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.bootstrap_accuracy["hamiltonian_train"] = accuracy_score(y_true,y_pred)
        ### hamiltonian_val accuracy 
        boolean_mask = self.hamiltonian_summary["type_of"]=="val"
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.bootstrap_accuracy["hamiltonian_val"] = accuracy_score(y_true,y_pred)
        ### hamiltonian_test accuracy 
        boolean_mask = np.logical_and(self.hamiltonian_summary["type_of"]=="test", np.in1d(self.hamiltonian_summary.phase, self.allowed_windings))
        y_true = self.hamiltonian_summary.phase[boolean_mask].values
        y_pred = self.hamiltonian_summary.pred_phase[boolean_mask].values
        self.bootstrap_accuracy["hamiltonian_test"] = accuracy_score(y_true,y_pred)
        ### saving to disk
        if save_to_disk:
            with open(os.path.join(self.simulation_dir, "bootstrap_accuracy.csv"), 'w') as f:  
                w = csv.writer(f)
                w.writerows(self.bootstrap_accuracy.items())

    def compute_mean_feature_importance(self, sort_importances= False, save_to_disk = False):
        """
        Computes feature importances using averages from all experiments
        """
        feature_importances = []
        for exp in tqdm(range(self.n_experiments), desc="mean feature importances"):
            current_model = load(os.path.join(self.simulation_dir, "model", str(exp) + ".joblib"))
            feature_importances.append(current_model.feature_importances_)
        mean_feature_importance = np.mean(feature_importances, axis = 0)
        if sort_importances:
            sorted_args = np.argsort(mean_feature_importance)[::-1] 
            sorted_feature_importance = mean_feature_importance[sorted_args]
            self.feature_importance = dict(zip(sorted_args, sorted_feature_importance))  ##Adding 1 so that features start at 1
            self.cumulative_feature_importance = dict(zip(sorted_args, np.cumsum(sorted_feature_importance)))
        else:
            non_sorted_args = np.arange(len(mean_feature_importance))
            self.feature_importance = dict(zip(non_sorted_args, mean_feature_importance))
            self.cumulative_feature_importance = dict(zip(non_sorted_args, np.cumsum(mean_feature_importance) ) )
        if save_to_disk:
            with open(os.path.join(self.simulation_dir, "feature_importance.csv"), 'w') as f:  
                w = csv.writer(f)
                w.writerows(self.feature_importance.items())
            with open(os.path.join(self.simulation_dir, "cumulative_feature_importance.csv"), 'w') as f:  
                w = csv.writer(f)
                w.writerows(self.cumulative_feature_importance.items())

    def create_plot(self, fig_params, xlabel_params, ylabel_params, title_params, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}):
        """
        Creates a figure for plotting.
         
        input
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        
        return
        figure: a figure with properly set axes for plotting.
        """
        figure = plt.figure(**fig_params)
        #if fit_axes:    
        #    plt.xlim(np.min(self.t2), np.max(self.t2))
        #    plt.ylim(np.min(self.t1), np.max(self.t1))
        if len(xlim_params) > 0:
            plt.xlim(**xlim_params)
        if len(ylim_params) > 0:
            plt.ylim(**ylim_params)
        if len(xticks_params) > 0:
            plt.xticks(**xticks_params)
        if len(yticks_params) > 0:
            plt.yticks(**yticks_params)
        if len(xlabel_params) > 0:
            plt.xlabel(**xlabel_params)
        if len(ylabel_params) > 0:
            plt.ylabel(**ylabel_params)
        if len(title_params) > 0:
            plt.title(**title_params)
        #plt.show()
        return figure   

    def imshow_winding_grid(self, winding, imshow_params, colorbar_params={}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Plots a heatmap of windings using imshow.

        input
        winding: an int. One of the allowed windings.
        imshow_params: a dict with parameters.
        colorbar_params: a dict with parameters for plt.colorbar.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.
        """
        if len(fig_params) > 0:
            figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        #figure = plt.figure(figsize=(10,10))
        winding_grid = self.winding_grid[winding]
        imshow_params["X"] = winding_grid
        plt.imshow(**imshow_params)
        #if len(colorbar_params) > 0:
        if len(colorbar_params) > 0:
            labelsize = None
            if "labelsize" in colorbar_params:
                labelsize = colorbar_params.pop("labelsize")
            cbar = plt.colorbar(**colorbar_params)
            if labelsize is not None:
                cbar.ax.tick_params(labelsize=labelsize)
                colorbar_params["labelsize"] = labelsize
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params) 

    def merge_imshow_winding_grids(self, winding_params, colorbar_params, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        """
        Produces a merged heatmap of windings with imshow.

        input
        winding_params: a dict of dicts. Each dict contains parameters for the imshow heatmap of a single winding.
        colorbar_params: a dict with parameters for plt.colorbar.
        fig_params: a dict with parameters for plt.fig.
        xlabel_params: a dict with parameters for plt.xlabel.
        ylabel_params: a dict with parameters for plt.ylabel.
        title_params: a dict with parameters for plt.title.
        savefig_params: a dict with parameters for plt.savefig. If empty, the plot is not saved.       
        """    
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        for winding in self.allowed_windings:
            if not winding in winding_params.keys():
                print("Skipping winding {}".format(str(winding)))
                continue
            print("Plotting winding {}".format(str(winding)))
            imshow_params = winding_params[winding]
            cb_params={}
            if winding in colorbar_params:
                cb_params = colorbar_params[winding]
            #print("cb_params: ", cb_params)

            ##############################################
            winding_grid = self.winding_grid[winding]
            imshow_params["X"] = winding_grid
            #print("unique values: ", np.unique(winding_grid))
            #print("max value: ", np.max(winding_grid))
            plt.imshow(**imshow_params)
            if len(cb_params) > 0:
                labelsize = None
                if "labelsize" in cb_params:
                    labelsize = cb_params.pop("labelsize")
                cbar = plt.colorbar(**cb_params)
                if "ticks" in cb_params:
                    cbar.set_ticks(cb_params["ticks"])
                if labelsize is not None:
                    cbar.ax.tick_params(labelsize=labelsize)
                    cb_params["labelsize"] = labelsize
            ##############################################
            #plt.tight_layout()
            #self.imshow_winding_grid(winding, imshow_params, colorbar_params = cb_params)
        #if len(colorbar_params) > 0:
        #    labelsize = None
        #    if "labelsize" in colorbar_params:
        #        labelsize = colorbar_params.pop("labelsize")
        #    cbar = plt.colorbar(**colorbar_params)
        #    if labelsize is not None:
        #        cbar.ax.tick_params(labelsize=labelsize)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)  

    def plot_feature_importances(self, n_features=None, plot = "bar", hist_precision = 1000, plot_params = {}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params = None, savefig_params={}):
        """
        Plots feature importances
        
        input
        n_features: number of features to show in plot. If None, all are showed
        """
        if n_features is None:
            n_features = len(self.feature_importance)
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        sorted_args = list(self.feature_importance.keys())[:n_features]
        importances = list(self.feature_importance.values())[:n_features]
        #print("sorted_args: ", sorted_args)
        #print("importances: ", importances)
        if plot == "bar":
            plot_params["x"] = sorted_args
            plot_params["height"] = importances
            plt.bar(**plot_params)
        elif plot == "hist":
            #weighted_args = []
            #for arg, imp in zip(sorted_args, importances):
            #    weighted_args = weighted_args + [arg]*int(imp*hist_precision)
            plot_params["x"] = sorted_args
            plot_params["weights"] = importances
            #plot_params["height"] = importances
            if "bins" not in plot_params:
                plot_params["bins"] = len(sorted_args)
                #plot_params["bins"] = 30
                plt.hist(**plot_params)
        if tight_params is not None:
                plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
        #plt.bar(importances)

    def plot_cumulative_feature_importances(self, n_features=None, plot = "bar", hist_precision = 1000, plot_params = {}, fig_params={}, xlabel_params={}, ylabel_params={}, title_params={}, xlim_params={}, ylim_params={}, xticks_params ={}, yticks_params={}, tight_params=None, savefig_params={}):
        if n_features is None:
            n_features = len(self.feature_importance)
        figure = self.create_plot(fig_params, xlabel_params, ylabel_params, title_params, xlim_params, ylim_params, xticks_params, yticks_params)
        #sorted_args = [str(ix) for ix in list(self.cumulative_feature_importance.keys())[:n_features]]
        sorted_args = list(self.cumulative_feature_importance.keys())[:n_features]
        importances = list(self.cumulative_feature_importance.values())[:n_features]
        #args = np.array(list(self.cumulative_feature_importance.keys())[:n_features])
        #sorted_ix = np.argsort(args)
        #sorted_args = args[sorted_ix]
        #importances = np.array(list(self.cumulative_feature_importance.values())[:n_features])[sorted_ix]
        #print("sorted_args: ", sorted_args)
        #print("importances: ", importances)
        if plot == "bar":
            plot_params["x"] = sorted_args
            plot_params["height"] = importances
            plt.bar(**plot_params)
        elif plot == "hist":
            #weighted_args = []
            #for arg, imp in zip(sorted_args, importances):
            #    weighted_args = weighted_args + [arg]*int(imp*hist_precision)
            plot_params["x"] = sorted_args
            plot_params["weights"] = importances
            #plot_params["height"] = importances
            if "bins" not in plot_params:
                plot_params["bins"] = len(sorted_args)
                #plot_params["bins"] = 30
                plt.hist(**plot_params)
        if tight_params is not None:
            plt.tight_layout(**tight_params)
        if len(savefig_params) > 0:
            plt.savefig(**savefig_params)
        
               

   # def plot_feature_importances(tree_clf, n_features = "all", return_arrays = False):
   # """
   # Plots a histogram of feature importances.
#
#    input
#    tree_clf: a tree based classifier
#    n_features: number of features to show in plot. If "all", all are showed
#    return_arrays: a boolean variable. Whether return importance arrays or not

#    return 
#    sorted_indices: array of features in decreasing order of importance
#    sorted_importances: array of importances in decreasing order
#    cumulative_importances: array of cumulative importances computed from sorted_importances
#    """
#    importances = tree_clf.feature_importances_
#    sorted_indices = np.argsort(importances)[::-1]
#    sorted_importances = importances[sorted_indices]
#    cumulative_importances = np.cumsum(sorted_importances)
#    sorted_features = np.array([str(ix) for ix in sorted_indices ])
#    if n_features == "all":
#        n_features = len(sorted_indices)
#    fig, ax = plt.subplots(2,1, figsize = (30,50))
#    for ix, feat in enumerate(sorted_features[:n_features]):
#        ax[0].bar(feat, sorted_importances[ix])
#        ax[0].set_title("Feature importances", size = 24)
#        ax[0].set_xlabel("Feature", fontsize = 24)
#        ax[0].set_ylabel("Importance", fontsize = 24)
#        ax[1].bar(feat, cumulative_importances[ix])
#        ax[1].set_title("Cumulative feature importances", size = 24)
#        ax[1].set_xlabel("Feature", fontsize = 24)
#        ax[1].set_ylabel("Cumulative importance ", size = 24)
   
#    plt.show()
#    if return_arrays:
#        return sorted_indices, sorted_importances, cumulative_importances
#    else:
#        return None


    
