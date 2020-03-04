#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:23:25 2020

@author: linneu
"""


import papermill as pm
from tqdm import tqdm
import os

WDIR = "/home/linneu/ml_topological_phases_in_real_space/0_preprocessing"
os.chdir(WDIR)

template = "preprocessing_template.ipynb"
parameters = {"allowed_windings":[0,1], "epsilon":0.01, "to_csv":True}
kernel_name = "ml_top_phases"
grid_folders_csv_names_outputs = [("/home/linneu/ssh_grids/ssh1/periodic_100_6561", "/home/linneu/ssh_csvs/ssh1/periodic_100_6561.csv","preprocessing_output_periodic_100_6561.ipynb"),\
                             ("/home/linneu/ssh_grids/ssh1/periodic_140_6561", "/home/linneu/ssh_csvs/ssh1/periodic_140_6561.csv","preprocessing_output_periodic_140_6561.ipynb"),\
                             ("/home/linneu/ssh_grids/ssh1/periodic_180_6561", "/home/linneu/ssh_csvs/ssh1/periodic_180_6561.csv","preprocessing_output_periodic_180_6561.ipynb"),\
                             ("/home/linneu/ssh_grids/ssh1/periodic_220_6561", "/home/linneu/ssh_csvs/ssh1/periodic_220_6561.csv","preprocessing_output_periodic_220_6561.ipynb")\
                             ]
    
def execute_notebook(template,output_file,parameters,kernel_name,grid_folder,csv_name):
    
    parameters["grid_folder"]=grid_folder
    parameters["csv_name"]=csv_name
    nb = pm.execute_notebook(template,
                        output_file,
                        parameters=parameters,
                        kernel_name=kernel_name)
    #nbs = []
    #nbs.append(nb)
    return nb

nbs = {}
for grid_folder, csv_name, output_file in tqdm(grid_folders_csv_names_outputs):
    filename = csv_name.split(".")[0].split("/")[-1]
    nbs[filename] = execute_notebook(template,output_file,parameters,kernel_name,grid_folder,csv_name)
    

    