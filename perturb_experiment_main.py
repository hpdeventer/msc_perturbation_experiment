import logging
import warnings
import time
import numpy as np
import gc
from concurrent.futures import ProcessPoolExecutor
from comparison_models import *
from perturb_experiment_helper import * 

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

start_time = time.time()

max_num_trials = 3
trial_chunks = 1

if __name__ == "__main__":
    optimizers = ['adam', 'sgd']
    pseudorehearsals = [True, False]
    input_dimensions = [x for x in range(1,7)]
    trial_numbers = list(range(max_num_trials))

    # Split the trial_numbers into 3 roughly equal parts
    subsets_of_trials = np.array_split(trial_numbers, trial_chunks)

    for trial_subset in subsets_of_trials:
        all_experiments = []

        # Create all combinations of parameters
        for trial in trial_subset:
            for optimizer in optimizers:
                for use_pseudorehearsal in pseudorehearsals:
                    for input_dimension in input_dimensions:
                        experiment_params = (trial, input_dimension, 20, 128, use_pseudorehearsal, "mse", optimizer, 0)
                        all_experiments.append(experiment_params)

        # Execute all experiments in parallel
        with ProcessPoolExecutor() as executor:
            executor.map(perturbation_experiment, *zip(*all_experiments))

        # Cleanup to save memory
        del all_experiments
        gc.collect()

    for optimizer in optimizers:
        for use_pseudorehearsal in pseudorehearsals:
            for input_dimension in input_dimensions:
                save_aggregated_data(input_dim=input_dimension,
                                    use_pseudorehearsal=use_pseudorehearsal,
                                    optimizer=optimizer,
                                    trials=max_num_trials,
                                    num_models=19)

    print(f"Execution Time: {time.time() - start_time} seconds")
