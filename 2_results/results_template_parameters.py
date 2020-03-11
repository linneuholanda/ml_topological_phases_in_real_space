#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:23:54 2020

@author: linneu
"""
import os
import numpy as np
from simulation_results import *

def get_ssh1_parameters(ssh_type, chain_length, n_hamiltonians, experiment_name="0", n_experiments=None, load_hamiltonian_summary=False, 
                        base_simulations_dir = "/home/linneu/ssh_simulations/ssh1", base_figs_dir="/home/linneu/ml_topological_phases_in_real_space/paper/ssh1"):
    
    if not os.path.isdir(base_figs_dir):
        os.mkdir(base_figs_dir)
    figs_dir = os.path.join(base_figs_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians))
    if not os.path.isdir(figs_dir):
        os.mkdir(figs_dir)    
    simulation_dir = os.path.join(base_simulations_dir,"{}_{}_{}".format(ssh_type,chain_length,n_hamiltonians)) 
    experiment = Experiment(simulation_dir, experiment_name)
    simulation = ExperimentEnsemble(simulation_dir, n_experiments, load_hamiltonian_summary)
    ### Defining parameters dict   
    parameters = {
    
    #########################################################################
    ################### Results from a single experiment ####################
    #########################################################################
   
    ### Visualizing train/val/test splits with scatter plot for a single experiment
    "train_val_test_fig_params": {"fig_params":{"figsize": (6,6)}}, 
    "train_val_test_val_params": {"marker": "s", "s": 64, "alpha": 0.2, "color": "salmon", "label": "val"},
    "train_val_test_test_params": {"marker": "s", "s": 64, "alpha": 0.5, "color": "royalblue", "label": "test"},
    "train_val_test_train_params": {"marker": "o", "s": 3, "alpha": 1, "color": "green", "label": "train"},
    "train_val_test_legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "train_val_test_xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "train_val_test_ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "train_val_test_title_params": {},
    "train_val_test_xlim_params": {"left": int(np.min(experiment.t2)), "right": int(np.max(experiment.t2))},
    "train_val_test_ylim_params": {"bottom": int(np.min(experiment.t1)), "top": int(np.max(experiment.t1))},
    "train_val_test_xticks_params": {"fontsize": 14},
    "train_val_test_yticks_params": {"fontsize": 14},
    "train_val_test_tight_params": {},
    "train_val_test_savefig_params": {"fname": os.path.join(figs_dir,"train_val_test_split_experiment_{}.png".format(experiment_name))},

    ### Plotting train winding labels with scatter plot for a single experiment
    "train_winding_fig_params": {"fig_params":{"figsize": (6,6)}}, 
    "train_winding_winding_params": {0: {"marker": "o", "s": 10, "color": "red", "label": 0}, \
                                     1: {"marker": "o", "s": 10, "color": "blue", "label": 1}, \
                                    },
    "train_winding_legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "train_winding_xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "train_winding_ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "train_winding_title_params": {},
    "train_winding_xlim_params": {"left": int(np.min(experiment.t2)), "right": int(np.max(experiment.t2))},
    "train_winding_ylim_params": {"bottom": int(np.min(experiment.t1)), "top": int(np.max(experiment.t1))},
    "train_winding_xticks_params": {"fontsize": 14},
    "train_winding_yticks_params": {"fontsize": 14},
    "train_winding_tight_params": {},
    #"train_winding_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/train_winding_labels_experiment_0.png",
    "train_winding_savefig_params": {"fname": os.path.join(figs_dir,"train_winding_labels_experiment_{}.png".format(experiment_name))},

    ### Plotting prediction grid with pcolormesh
    "prediction_grid_fig_params": {"fig_params":{"figsize": (6,6)}},
    "prediction_grid_winding_params": {0: {"marker": "o", "s": 10, "color": "red", "label": 0}, 
                                      1: {"marker": "o", "s": 10, "color": "blue", "label": 1}},
    #"prediction_grid_pcolormesh_params": {"cmap": ListedColormap(["hotpink", "lightskyblue"]), "alpha": 0.5},
    "prediction_grid_pcolormesh_params": {"cmap": "ssh1", "alpha": 0.5},
    "prediction_grid_legend_params": {"framealpha": 0.8, "loc": "upper right", "fontsize": 16},
    "prediction_grid_xlabel_params": {"xlabel": "$t_2$", "fontsize": 24},
    "prediction_grid_ylabel_params": {"ylabel": "$t_1$", "fontsize": 24},
    "prediction_grid_title_params": {},
    "prediction_grid_xlim_params": {"left": int(np.min(experiment.t2)), "right": int(np.max(experiment.t2))},
    "prediction_grid_ylim_params": {"bottom": int(np.min(experiment.t1)), "top": int(np.max(experiment.t1))},
    "prediction_grid_xticks_params": {"fontsize": 14},
    "prediction_grid_yticks_params": {"fontsize": 14},
    "prediction_grid_tight_params": {},
    #"prediction_grid_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/prediction_grid_experiment_0.png".format(ssh_type,chain_length,n_hamiltonians),
    "prediction_grid_savefig_params": {"fname": os.path.join(figs_dir,"prediction_grid_experiment_{}.png".format(experiment_name))},

    ### Plotting Winding Heatmaps for a single experiment
    "winding_heatmap_winding": 1,
    "winding_heatmap_imshow_params": {"cmap": "bwr_r", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2],\
                                     "vmin": 0, "vmax":1},
    "winding_heatmap_colorbar_params": {"mappable": None, "labelsize": 24},
    "winding_heatmap_fig_params": {"figsize": (12,12)},
    "winding_heatmap_xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "winding_heatmap_ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "winding_heatmap_title_params": {},
    "winding_heatmap_xlim_params": {"left": int(np.min(experiment.t2)), "right": int(np.max(experiment.t2))},
    "winding_heatmap_ylim_params": {"bottom": int(np.min(experiment.t1)), "top": int(np.max(experiment.t1))},
    "winding_heatmap_xticks_params": {"fontsize": 24},
    "winding_heatmap_yticks_params": {"fontsize": 24},
    "winding_heatmap_tight_params": {},
    "winding_heatmap_savefig_params": {"fname": os.path.join(figs_dir,"winding_1_grid_experiment_{}.png".format(experiment_name))},

    ##################################################################################
    ################### Bootstrapped results from all experiments ####################
    ##################################################################################
    
    ### Plotting simulation winding heatmaps
    "sim_winding_heatmap_winding_params": {1: {"cmap": "bwr_r", "aspect": "equal", "alpha": None, "origin": "lower", "extent": [-2,2,-2,2], \
                                           "vmin": 0, "vmax": 1}},
    "sim_winding_heatmap_colorbar_params": {1: {"mappable": None, "labelsize": 24, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0], "pad": 0.1, "shrink": 0.8, \
                                           "extend": "neither"}},
    "sim_winding_heatmap_fig_params": {"figsize": (12,12)},
    "sim_winding_heatmap_xlabel_params": {"xlabel": "$t_2$", "fontsize": 48},
    "sim_winding_heatmap_ylabel_params": {"ylabel": "$t_1$", "fontsize": 48},
    "sim_winding_heatmap_title_params": {},
    "sim_winding_heatmap_xlim_params": {"left": int(np.min(simulation.t2)), "right": int(np.max(simulation.t2))},
    "sim_winding_heatmap_ylim_params": {"bottom": int(np.min(simulation.t1)), "top": int(np.max(simulation.t1))},
    "sim_winding_heatmap_xticks_params": {"fontsize": 24},
    "sim_winding_heatmap_yticks_params": {"fontsize": 24},
    "sim_winding_heatmap_tight_params": {},
    #"sim_winding_heatmap_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/simulation_merged_winding_grid.png".format(ssh_type,chain_length,n_hamiltonians),
    "sim_winding_heatmap_savefig_params": {"fname": os.path.join(figs_dir,"bootstrapped_merged_winding_grid.png")},

    ### Plotting feature importances
    "feature_importances_n_features": None,
    "feature_importances_plot_type": "bar",
    "feature_importances_plot_params": {"color": "indianred", "width": 0.7},
    "feature_importances_hist_precision": 1000,
    "feature_importances_fig_params": {"figsize": (12,12)}, 
    "feature_importances_xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "feature_importances_ylabel_params": {"ylabel": "reduction in information entropy (%)", "fontsize": 24},
    "feature_importances_title_params": {"label": "Information entropy signature - SSH 1", "fontsize": 24},
    "feature_importances_xlim_params": {},
    "feature_importances_ylim_params": {},
    "feature_importances_xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "feature_importances_yticks_params": {"fontsize": 24},
    "feature_importances_tight_params": {},
    #"feature_importances_path_to_save": "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/{}_{}_{}/feature_importances.png".format(ssh_type,chain_length,n_hamiltonians),
    "feature_importances_savefig_params": {"fname": os.path.join(figs_dir,"bootstrapped_feature_importances.png")},

    ### Plotting cumulative feature importances
    "cumulative_features_n_features": None,
    "cumulative_features_plot_type": "bar",
    "cumulative_features_hist_precision": 1000,
    "cumulative_features_plot_params": {"color":"indianred", "width": 0.7},
    "cumulative_features_fig_params": {"figsize": (12,12)},
    "cumulative_features_xlabel_params": {"xlabel": "lattice site", "fontsize": 24},
    "cumulative_features_ylabel_params": {"ylabel": "cumulative reduction in information entropy (%)", "fontsize": 24},
    "cumulative_features_title_params": {"label": "Cumulative information entropy signature - SSH 1", "fontsize": 24},
    "cumulative_features_xlim_params": {},
    "cumulative_features_ylim_params": {},
    "cumulative_features_xticks_params": {"ticks": [int(i) for i in np.linspace(0,int(chain_length)-1,10).astype(int)], "fontsize": 24},
    "cumulative_features_yticks_params": {"fontsize": 24},
    "cumulative_features_tight_params": {},
    #cumulative_features_path_to_save = "/home/linneu/ml_topological_phases_in_real_space/paper/ssh1/periodic_100_6561/cumulative_feature_importances.png"
    "cumulative_features_savefig_params": {"fname": os.path.join(figs_dir,"bootstrapped_cumulative_feature_importances.png")},
    }
    return parameters, experiment, simulation
