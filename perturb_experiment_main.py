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
    trial_numbers = list(range(2))
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]

    for input_dim in range(1, 3):
        for partition in range(1, 4):
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
