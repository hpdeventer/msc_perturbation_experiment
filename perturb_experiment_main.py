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

# Constants see perturb_experiment_main.py
    
#   experiment_arguments = {
#       'trial_number': 1,
#       'input_dimension': 6,
#       'partition_number': 10,
#       'use_pseudorehearsal': True,
#       'loss_function': 'mae',
#       'verbose': 0.,
#       'optimizer': 'adam'
#   }

# Execute the experiment
#basic_perturbation_experiment(**experiment_arguments)
'''
import time
from multiprocessing import Pool
from datetime import datetime

def wrapper(args):
    """
    A wrapper function to unpack the dictionary of arguments and call the original function.
    """
    return basic_perturbation_experiment(**args)

def create_progress_file(input_dim, partition, start_timestamp, end_timestamp):
    """
    Create a txt file with timestamps indicating that the experiments for a particular 
    input dimension and partition have been completed.
    """
    formatted_start_time = start_timestamp.strftime('%Y%m%d_%H%M%S')
    formatted_end_time = end_timestamp.strftime('%Y%m%d_%H%M%S')
    filename = f"completed_input_dim_{input_dim}_partition_{partition}_{formatted_end_time}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Experiments for input dimension {input_dim} and partition {partition} started at {formatted_start_time} and completed at {formatted_end_time}.")

def parallel_execute():
    # List of trial numbers, optimizers, and boolean values for use_pseudorehearsal
    trial_numbers = list(range(30))
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]

    for input_dim in range(1, 6):
        for partition in range(1, 10):
            # Capture the start timestamp for this combination of input_dim and partition
            start_timestamp = datetime.now()
            
            # Create a list of all combinations of arguments for current input_dim and partition
            experiments = []
            for trial in trial_numbers:
                for optimizer in optimizers:
                    for use_pseudorehearsal in pseudorehearsals:
                        experiment_arguments = {
                            'trial_number': trial,
                            'input_dimension': input_dim,
                            'partition_number': partition,
                            'use_pseudorehearsal': use_pseudorehearsal,
                            'loss_function': 'mae',
                            'verbose': 0,
                            'optimizer': optimizer
                        }
                        experiments.append(experiment_arguments)

            # Use multiprocessing to execute the function in parallel for the current subset of experiments
            with Pool() as pool:
                pool.map(wrapper, experiments)
            
            # Capture the end timestamp for this combination of input_dim and partition
            end_timestamp = datetime.now()
            
            # After all trials for a specific input dimension and partition are completed, create a progress file with timestamps
            create_progress_file(input_dim, partition, start_timestamp, end_timestamp)

if __name__ == "__main__":
    overall_start_time = time.time()

    parallel_execute()

    print(f"Total Execution Time: {time.time() - overall_start_time} seconds")
'''

"""import time
import numpy as np
import os
from multiprocessing import Pool
import gc"""

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

def process_combination(args):
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

    with ProcessPoolExecutor() as executor:
        executor.map(wrapper, experiments)

    aggregate_and_save(input_dim, partition, optimizers, pseudorehearsals, trial_numbers)


"""def process_combination(args):
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

    with Pool() as pool:
        pool.map(wrapper, experiments)

    aggregate_and_save(input_dim, partition, optimizers, pseudorehearsals, trial_numbers)"""

"""if __name__ == "__main__":
    start_time = time.time()

    trial_numbers = list(range(3))
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]
    combinations = [(input_dim, partition, trial_numbers, optimizers, pseudorehearsals)
                    for input_dim in range(1, 3) #7
                    for partition in range(1, 3)] #11

    with Pool() as pool:
        pool.map(process_combination, combinations)

    print(f"Execution Time: {time.time() - start_time} seconds")"""


if __name__ == "__main__":
    start_time = time.time()

    trial_numbers = list(range(3)) # 31
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]
    combinations = [(input_dim, partition, trial_numbers, optimizers, pseudorehearsals)
                    for input_dim in range(1, 3) #7
                    for partition in range(1, 3)] #11

    with ProcessPoolExecutor() as executor:
        executor.map(process_combination, combinations)

    print(f"Execution Time: {time.time() - start_time} seconds")
