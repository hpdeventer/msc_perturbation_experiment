# Standard library imports
#import os
import logging
import warnings
#import gc
#import time

# Third-party library imports
#import numpy as np
#import tensorflow as tf

# Local module imports
from comparison_models import *
from perturb_experiment_helper import *

# Configuration for logging and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import time
import numpy as np
import os
import gc
from concurrent.futures import ProcessPoolExecutor


def wrapper(args):
    basic_perturbation_experiment(**args)

def load_and_save_data(input_dim=2, 
                       partition=4, 
                       use_pseudorehearsal=False, 
                       optimizer='sgd', 
                       trials=30, 
                       num_models=6):
    """
    Load and save data from specified trials and models to a folder.
    
    Parameters:
    - input_dim (int): Input dimension.
    - partition (int): Partition number.
    - use_pseudorehearsal (bool): Whether pseudo rehearsal is used.
    - optimizer (str): Type of optimizer.
    - trials (int): Number of trials.
    - num_models (int): Number of models.
    """

    base_folder = f"results/input_dim_{input_dim}_partition_num_{partition}_\
{use_pseudorehearsal}_{optimizer}"

    # Initialize the main data storage with the desired structure
    data = {
        "min_distance": [],
        "max_distance": []
    }

    for j in range(num_models):
        data[f"model_{j}_perturbations"] = []

    for i in range(trials):
        trial_folder = f"{base_folder}/trial_{i}"

        min_distance_path = f"{trial_folder}/distances/min_distances.npy"
        max_distance_path = f"{trial_folder}/distances/max_distances.npy"

        # Load the distances from the numpy files
        if os.path.exists(min_distance_path):
            min_distance = np.load(min_distance_path)
            data["min_distance"].append(min_distance)
        else:
            print(f"Warning: {min_distance_path} not found.")

        if os.path.exists(max_distance_path):
            max_distance = np.load(max_distance_path)
            data["max_distance"].append(max_distance)
        else:
            print(f"Warning: {max_distance_path} not found.")

        for j in range(num_models):
            perturbation_path = f"{trial_folder}/perturbations/model_{j}/absolute_perturbation.npy"

            # Load the perturbation from the numpy file
            if os.path.exists(perturbation_path):
                perturbation_data = np.load(perturbation_path)
                data[f"model_{j}_perturbations"].append(perturbation_data)
            else:
                print(f"Warning: {perturbation_path} not found.")

    # Convert lists of numpy arrays to a single concatenated numpy array
    for key in data:
        if data[key]:  # Check if the list is not empty
            data[key] = np.concatenate(data[key], axis=0).flatten()
        else:
            data[key] = None  # Set the value to None if the list is empty

    # Save the data
    save_folder = "aggregated_results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = f"{save_folder}/data_input_dim_{input_dim}_partition_{partition}.npy"
    np.save(save_path, data)

def prepare_experiments(args):
    input_dim, partition, trial_numbers, optimizers, pseudorehearsals = args
    experiments = []
    for trial in trial_numbers:
        for optimizer in optimizers:
            for use_pseudorehearsal in pseudorehearsals:
                experiment_arguments = {
                    'trial_number': trial,
                    'input_dimension': input_dim,
                    'partition_number': partition,
                    'use_pseudorehearsal': use_pseudorehearsal,
                    'loss_function': 'mse',
                    'verbose': 0,
                    'optimizer': optimizer
                }
                experiments.append(experiment_arguments)
    return experiments

if __name__ == "__main__":
    start_time = time.time()

    trial_numbers = list(range(31))
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]

    for partition in range(1, 11): # Iterate over partition numbers using a for-loop
        input_dim_combinations = [(input_dim, partition, trial_numbers, optimizers, pseudorehearsals) 
                                  for input_dim in range(1, 7)]

        all_experiments = []
        for combo in input_dim_combinations:
            all_experiments.extend(prepare_experiments(combo))

        # Execute all experiments for the given partition in parallel
        with ProcessPoolExecutor() as executor:
            executor.map(wrapper, all_experiments)

        # Aggregate results after all experiments for the given partition have run
        for combo in input_dim_combinations:
            input_dim, _, _, _, _ = combo
            
            # Loop through optimizers and pseudorehearsals to call the `load_and_save_data` for each combination
            for optimizer in optimizers:
                for use_pseudorehearsal in pseudorehearsals:
                    load_and_save_data(input_dim=input_dim, 
                                       partition=partition, 
                                       optimizer=optimizer, 
                                       use_pseudorehearsal=use_pseudorehearsal, 
                                       trials=len(trial_numbers), 
                                       num_models=6)
                    
        # Cleanup to save memory
        del all_experiments
        gc.collect()

    print(f"Execution Time: {time.time() - start_time} seconds")

