#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:23:54 2020

@author: linneu
"""
import os
import numpy as np
from simulation_results import *

def get_parameters_ssh1(ssh_type, chain_length, n_hamiltonians, experiment_name="0", n_experiments=None, load_hamiltonian_summary=False, base_simulations_dir="/home/rio/ssh_simulations/ssh1",
                        base_figs_dir="/home/rio/ml_topological_phases_in_real_space/paper/ssh1"):
    
    if not os.path.isdir(base_figs_dir):
        os.mkdir(base_figs_dir)
    figs_dir = os.path.join(base_figs_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians))
    if not os.path.isdir(figs_dir):
        os.mkdir(figs_dir)    
    simulation_dir = os.path.join(base_simulations_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians)) 
    
    #########################################################################
    ################### Results from a single experiment ####################
    #########################################################################
    
    parameters = { 
    ### Visualizing train/val/test splits with scatter_train_val_test
    "scatter_train_val_test_params": {
    "fig_params":{"figsize": [6,6]}, 
    "val_params": {"marker": "s", "s": 64, "alpha": 0.2, "color": "salmon", "label": "val"},
    "test_params": {"marker": "s", "s": 64, "alpha": 0.5, "color": "royalblue", "label": "test"},
    "train_params": {"marker": "o", "s": 3, "alpha": 1, "color": "green", "label": "train"},
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"scatter_train_val_test_experiment_{}.png".format(experiment_name))},
     },

    ### Plotting train winding labels with scatter_winding_train
    "scatter_winding_train_params": {
    "fig_params":{"figsize": [6,6]}, 
    "winding_params": {0: {"marker": "o", "s": 10, "color": "red", "label": 0}, \
                                     1: {"marker": "o", "s": 10, "color": "blue", "label": 1}, \
                                    },
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    #"train_winding_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/train_winding_labels_experiment_0.png",
    "savefig_params": {"fname": os.path.join(figs_dir,"scatter_winding_train_experiment_{}.png".format(experiment_name))},
     },
        
    ### Plotting prediction grid with pcolormesh_prediction_grid
    "pcolormesh_prediction_grid_params": {
    "fig_params": {"figsize": [6,6]},
    "winding_params": {0: {"marker": "o", "s": 10, "color": "red", "label": 0}, 
                                      1: {"marker": "o", "s": 10, "color": "blue", "label": 1}},
    #"prediction_grid_pcolormesh_params": {"cmap": ListedColormap(["hotpink", "lightskyblue"]), "alpha": 0.5},
    "pcolormesh_params": {"cmap": "ssh1", "alpha": 0.5},
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"pcolormesh_prediction_grid_experiment_{}.png".format(experiment_name))},
     },
    
    ### Plotting Winding Heatmaps with imshow_winding_grid
    "winding_cmap_dict":{0: "Reds", 1:"Blues"},
    "imshow_winding_grid_params": {
    "winding": 1,
    "imshow_params": {"cmap": "bwr_r", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2],\
                                     "vmin": 0, "vmax":1},
    #"colorbar_params": {"mappable": None, "labelsize": 24},
    "colorbar_params": {"mappable": None, "labelsize": 24, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0], "pad": 0.1, "shrink": 0.8, \
                                           "extend": "neither"},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"imshow_winding_grid_winding_{}_{}.png")}, #.format something
    }, 
        
    ##################################################################################
    ################### Bootstrapped results from all experiments ####################
    ##################################################################################
    
    ### Plotting simulation winding heatmaps with merge_imshow_winding_grids
    "merge_imshow_winding_grids_params": {
    #"winding_params": {1: {"cmap": "bwr_r", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2], \
    #                                       "vmin": 0, "vmax": 1}},
    "winding_params": {0: {"cmap": "Reds_alpha", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2], \
                                           "vmin": 0, "vmax": 1},
                       1: {"cmap": "Blues_alpha", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2], \
                                           "vmin": 0, "vmax": 1}},
    #"colorbar_params": {1: {"mappable": None, "labelsize": 24, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0], "pad": 0.1, "shrink": 0.8, \
    #                                       "extend": "neither"}},
    "colorbar_params": {},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #"sim_winding_heatmap_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/simulation_merged_winding_grid.png".format(ssh_type,chain_length,n_hamiltonians),
    #"savefig_params": {"fname": os.path.join(figs_dir,"merge_imshow_winding_grids_{}.png")},
    "savefig_params": {"fname": os.path.join(figs_dir,"merge_imshow_winding_grids_second_{}.png")},
     },
        
    ### Plotting feature importances with plot_feature_importances
    "plot_feature_importances_params": {
    "n_features": None,
    "plot": "bar",
    "plot_params": {"color": "indianred", "width": 0.7},
    "hist_precision": 1000,
    "fig_params": {"figsize": [12,12]}, 
    "xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "ylabel_params": {"ylabel": "reduction in information entropy (%)", "fontsize": 24},
    "title_params": {"label": "Information entropy signature - SSH 1", "fontsize": 24},
    "xlim_params": {},
    "ylim_params": {},
    "xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #"feature_importances_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/feature_importances.png".format(ssh_type,chain_length,n_hamiltonians),
    "savefig_params": {"fname": os.path.join(figs_dir,"plot_feature_importances.png")},
     },
    
    ### Plotting cumulative feature importances with plot_cumulative_feature_importances
    "plot_cumulative_feature_importances_params": {
    "n_features": None,
    "plot": "bar",
    "hist_precision": 1000,
    "plot_params": {"color":"indianred", "width": 0.7},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "ylabel_params": {"ylabel": "cumulative reduction in information entropy (%)", "fontsize": 24},
    "title_params": {"label": "Cumulative information entropy signature - SSH 1", "fontsize": 24},
    "xlim_params": {},
    "ylim_params": {},
    "xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #cumulative_features_path_to_save = "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/cumulative_feature_importances.png"
    "savefig_params": {"fname": os.path.join(figs_dir,"plot_cumulative_feature_importances.png")},
     },
    }   
    parameters["simulation_dir"] = simulation_dir
    parameters["experiment_name"] = experiment_name
    parameters["n_experiments"] = n_experiments
    parameters["load_hamiltonian_summary"] = load_hamiltonian_summary
    return parameters

def get_parameters_ssh2(ssh_type, chain_length, n_hamiltonians, experiment_name="0", n_experiments=None, load_hamiltonian_summary=False, base_simulations_dir="/home/rio/ssh_simulations/ssh2",
                        base_figs_dir="/home/rio/ml_topological_phases_in_real_space/paper/ssh2"):
    
    if not os.path.isdir(base_figs_dir):
        os.mkdir(base_figs_dir)
    figs_dir = os.path.join(base_figs_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians))
    if not os.path.isdir(figs_dir):
        os.mkdir(figs_dir)    
    simulation_dir = os.path.join(base_simulations_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians)) 
    
    #########################################################################
    ################### Results from a single experiment ####################
    #########################################################################
    
    parameters = { 
    ### Visualizing train/val/test splits with scatter_train_val_test
    "scatter_train_val_test_params": {
    "fig_params":{"figsize": [6,6]}, 
    "val_params": {"marker": "s", "s": 64, "alpha": 0.2, "color": "salmon", "label": "val"},
    "test_params": {"marker": "s", "s": 64, "alpha": 0.5, "color": "royalblue", "label": "test"},
    "train_params": {"marker": "o", "s": 3, "alpha": 1, "color": "green", "label": "train"},
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"scatter_train_val_test_experiment_{}.png".format(experiment_name))},
     },

    ### Plotting train winding labels with scatter_winding_train
    "scatter_winding_train_params": {
    "fig_params":{"figsize": [6,6]}, 
    "winding_params": {-1: {"marker": "o", "s": 10, "color": "red", "label": -1}, 
                        0: {"marker": "o", "s": 10, "color": "darkorange", "label": 0}, \
                        1: {"marker": "o", "s": 10, "color": "green", "label": 1}, \
                        2: {"marker": "o", "s": 10, "color": "blue", "label": 2}},
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    #"train_winding_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/train_winding_labels_experiment_0.png",
    "savefig_params": {"fname": os.path.join(figs_dir,"scatter_winding_train_experiment_{}.png".format(experiment_name))},
     },
        
    ### Plotting prediction grid with pcolormesh_prediction_grid
    "pcolormesh_prediction_grid_params": {
    "fig_params": {"figsize": [6,6]},
    "winding_params": {-1: {"marker": "o", "s": 10, "color": "red", "label": -1}, 
                        0: {"marker": "o", "s": 10, "color": "darkorange", "label": 0}, \
                        1: {"marker": "o", "s": 10, "color": "green", "label": 1}, \
                        2: {"marker": "o", "s": 10, "color": "blue", "label": 2}},
    #"prediction_grid_pcolormesh_params": {"cmap": ListedColormap(["hotpink", "lightskyblue"]), "alpha": 0.5},
    "pcolormesh_params": {"cmap": "ssh2", "alpha": 0.5},
    "legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 14},
    "yticks_params": {"fontsize": 14},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"pcolormesh_prediction_grid_experiment_{}.png".format(experiment_name))},
     },
    
    ### Plotting Winding Heatmaps with imshow_winding_grid
    "winding_cmap_dict":{-1:"Reds", 0: "Oranges", 1:"Greens", 2:"Blues"},
    "imshow_winding_grid_params": {
    "winding": -1,
    "imshow_params": {"cmap": "Reds", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2],\
                                     "vmin": 0, "vmax":1},
    #"colorbar_params": {"mappable": None, "labelsize": 24},
    "colorbar_params": {"mappable": None, "labelsize": 24, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0], "pad": 0.1, "shrink": 0.8, \
                                           "extend": "neither"},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    "savefig_params": {"fname": os.path.join(figs_dir,"imshow_winding_grid_winding_{}_{}.png")}, #.format something
    }, 
        
    ##################################################################################
    ################### Bootstrapped results from all experiments ####################
    ##################################################################################
    
    ### Plotting simulation winding heatmaps with merge_imshow_winding_grids
    "merge_imshow_winding_grids_params": {
    "winding_params": {-1: {"cmap": "Reds_alpha", "aspect": "equal", "alpha": 1, "origin": "lower", "extent": [-2,2,-2,2], \
                            "vmin": 0, "vmax": 1},
                        0: {"cmap": "Oranges_alpha", "aspect": "equal", "alpha": 1, "origin": "lower", "extent": [-2,2,-2,2], \
                            "vmin": 0, "vmax": 1},
                        1: {"cmap": "Greens_alpha", "aspect": "equal", "alpha": 1, "origin": "lower", "extent": [-2,2,-2,2], \
                            "vmin": 0, "vmax": 1},
                        2: {"cmap": "Blues_alpha", "aspect": "equal", "alpha": 1, "origin": "lower", "extent": [-2,2,-2,2],\
                           "vmin": 0, "vmax": 1}},
    #"colorbar_params": {1: {"mappable": None, "labelsize": 24, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0], "pad": 0.1, "shrink": 0.8, \                                    "extend": "neither"}},
    "colorbar_params": {},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "title_params": {},
    "xlim_params": {"left": -2, "right": 2},
    "ylim_params": {"bottom": -2, "top": 2},
    "xticks_params": {"fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #"sim_winding_heatmap_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/simulation_merged_winding_grid.png".format(ssh_type,chain_length,n_hamiltonians),
    "savefig_params": {"fname": os.path.join(figs_dir,"merge_imshow_winding_grids_{}.png")},
     },
        
    ### Plotting feature importances with plot_feature_importances
    "plot_feature_importances_params": {
    "n_features": None,
    "plot": "bar",
    "plot_params": {"color": "cadetblue", "width": 0.7},
    "hist_precision": 1000,
    "fig_params": {"figsize": [12,12]}, 
    "xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "ylabel_params": {"ylabel": "reduction in information entropy (%)", "fontsize": 24},
    "title_params": {"label": "Information entropy signature - SSH 2", "fontsize": 24},
    "xlim_params": {},
    "ylim_params": {},
    "xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #"feature_importances_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/feature_importances.png".format(ssh_type,chain_length,n_hamiltonians),
    "savefig_params": {"fname": os.path.join(figs_dir,"plot_feature_importances.png")},
     },
    
    ### Plotting cumulative feature importances with plot_cumulative_feature_importances
    "plot_cumulative_feature_importances_params": {
    "n_features": None,
    "plot": "bar",
    "hist_precision": 1000,
    "plot_params": {"color":"cadetblue", "width": 0.7},
    "fig_params": {"figsize": [12,12]},
    "xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "ylabel_params": {"ylabel": "cumulative reduction in information entropy (%)", "fontsize": 24},
    "title_params": {"label": "Cumulative information entropy signature - SSH 2", "fontsize": 24},
    "xlim_params": {},
    "ylim_params": {},
    "xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "yticks_params": {"fontsize": 24},
    "tight_params": {},
    #cumulative_features_path_to_save = "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/cumulative_feature_importances.png"
    "savefig_params": {"fname": os.path.join(figs_dir,"plot_cumulative_feature_importances.png")},
     },
    }   
    parameters["simulation_dir"] = simulation_dir
    parameters["experiment_name"] = experiment_name
    parameters["n_experiments"] = n_experiments
    parameters["load_hamiltonian_summary"] = load_hamiltonian_summary
    return parameters
