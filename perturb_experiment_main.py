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

def aggregate_and_save(input_dim, partition, optimizers, pseudorehearsals, trial_numbers):
    aggregated_results = {}

    for trial in trial_numbers:
        for optimizer in optimizers:
            for use_pseudorehearsal in pseudorehearsals:
                base_folder = f"results/input_dim_{input_dim}_partition_num_{partition}_{use_pseudorehearsal}_{optimizer}/trial_{trial}"

                key = f"{input_dim}_{partition}_{use_pseudorehearsal}_{optimizer}"
                if key not in aggregated_results:
                    aggregated_results[key] = {}
                aggregated_results[key][trial] = {}
                
                for i in range(partition):
                    for data_name, file_name in [('model_output_prior_training', 'output_prior_training.npy'),
                                                 ('model_output_after_training', 'output_after_training.npy'),
                                                 ('absolute_perturbation', 'absolute_perturbation.npy')]:
                        data_path = f"{base_folder}/model_outputs/model_{i}/{file_name}"
                        if os.path.exists(data_path):
                            aggregated_results[key][trial][data_name] = np.load(data_path)
                
                for data_name, file_name in [('min_distances', 'min_distances.npy'),
                                             ('max_distances', 'max_distances.npy')]:
                    data_path = f"{base_folder}/distances/{file_name}"
                    if os.path.exists(data_path):
                        aggregated_results[key][trial][data_name] = np.load(data_path)

                # Clear memory for this iteration
                del base_folder

    if not os.path.exists("aggregated_results"):
        os.makedirs("aggregated_results")
    np.save(f"aggregated_results/data_input_dim_{input_dim}_partition_{partition}.npy", aggregated_results)
    
    # Clear memory
    del aggregated_results
    gc.collect()


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
            aggregate_and_save(input_dim, partition, optimizers, pseudorehearsals, trial_numbers)

        # Cleanup to save memory
        del all_experiments
        gc.collect()

    print(f"Execution Time: {time.time() - start_time} seconds")

