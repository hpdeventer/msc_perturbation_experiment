# Standard library imports
import os
from typing import Dict, Union
#import logging
#import warnings
import gc

# Third-party library imports
import numpy as np
import tensorflow as tf

# Local module imports
from comparison_models import *

# Configuration for logging and warnings
#logging.getLogger('tensorflow').setLevel(logging.ERROR)
#warnings.filterwarnings('ignore')

# Constants
NUM_TEST_POINTS = 5_000
NUM_TRAIN_POINTS = 5_000 # must be more than pseudo_samples
NUM_PSEUDO_SAMPLES = 3500


def load_data_create_dict(input_dim: int, 
                          use_pseudorehearsal: bool , 
                          optimizer: str , 
                          trials: int , 
                          num_models: int) -> Dict[str, Union[np.ndarray, None]]:
    """
    Load data for specified trials and models from a structured directory and 
    return it as a dictionary.
    
    Parameters:
    ----------
    input_dim : int
        Input dimension.
    use_pseudorehearsal : bool
        Whether pseudo rehearsal is used.
    optimizer : str
        Type of optimizer ('sgd', etc.).
    trials : int
        Number of trials.
    num_models : int
        Number of models.

    Returns:
    -------
    dict
        A dictionary containing concatenated min and max distances and model 
        perturbations for each trial and model.
    """
    
    base_folder = f"results/input_dim_{input_dim}_{use_pseudorehearsal}_{optimizer}"

    # Initialize the dictionary with keys for min_distance, max_distance, and model perturbations
    data = {
        "min_distance": [],
        "max_distance": [],
        **{f"model_{j}_perturbations": [] for j in range(num_models)}
    }

    for i in range(trials):
        trial_folder = f"{base_folder}/trial_{i}"

        paths = {
            "min_distance": f"{trial_folder}/distances/min_distances.npy",
            "max_distance": f"{trial_folder}/distances/max_distances.npy",
            **{f"model_{j}_perturbations": f"{trial_folder}/perturbations/model_{j}/absolute_perturbation.npy" for j in range(num_models)}
        }

        for key, path in paths.items():
            if os.path.exists(path):
                data[key].append(np.load(path))
            else:
                print(f"Warning: {path} not found.")

    # Convert lists of numpy arrays to a single concatenated numpy array
    for key, value in data.items():
        data[key] = np.concatenate(value, axis=0).flatten() if value else None
                
    return data

def save_aggregated_data(input_dim: int, 
                         use_pseudorehearsal: bool, 
                         optimizer: str, 
                         trials: int, 
                         num_models: int) -> None:
    """
    Save aggregated data to a numpy file for specified parameters.

    The function aggregates data based on the provided parameters and 
    saves it in a structured directory named 'aggregated_results'.

    Parameters:
    ----------
    input_dim : int
        Input dimension.
    use_pseudorehearsal : bool
        Whether pseudo rehearsal is used.
    optimizer : str
        Type of optimizer ('sgd', etc.).
    trials : int
        Number of trials. Aggregation is done over 0 to this parameter the (max) number of trials
    num_models : int
        Number of models.

    Returns:
    -------
    None
    """
    
    base_folder = f"results/input_dim_{input_dim}_{use_pseudorehearsal}_{optimizer}"

    # Save the data
    save_folder = "aggregated_results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = f"{save_folder}/input_dim_{input_dim}_{use_pseudorehearsal}_{optimizer}.npy"
    
    # Fetch the aggregated data
    data = load_data_create_dict(input_dim, use_pseudorehearsal, optimizer, trials, num_models)
    
    # Save the data to the specified path
    np.save(save_path, data)

def pseudorehearsal(input_dim: int, num_samples: int, 
                    model: tf.keras.Model, 
                    train_x: np.ndarray, 
                    train_y: np.ndarray, 
                    seed_val: int) -> tuple:
    '''
    Generate pseudorehearsal samples using a given model, combine them with the original training data, 
    and return the shuffled combined dataset.

    Args:
    - input_dim (int): Dimension of the input space.
    - num_samples (int): Number of pseudorehearsal samples to generate.
    - model (tf.keras.Model): Model to generate rehearsal targets.
    - train_x (np.ndarray): Original training input data.
    - train_y (np.ndarray): Original training target data.
    - seed_val (int): Seed value for random number generation.

    Returns:
    - tuple: Shuffled combined input and target data.
    '''
    
    # Generate pseudorehearsal samples, assume uniform over [0,1]^n
    rng = np.random.default_rng(seed_val)
    rehearsal_samples = rng.uniform(0, 1, size=(num_samples, input_dim))
    #print("rehearsal starts")
    rehearsal_targets = model(rehearsal_samples)
    #print("ends")
    
    # Combine training data with pseudorehearsal data and shuffle them
    combined_x = np.concatenate([train_x, rehearsal_samples], axis=0)
    combined_y = np.concatenate([train_y, rehearsal_targets], axis=0)
    
    indices = np.arange(combined_x.shape[0])
    rng.shuffle(indices)

    shuffled_x = combined_x[indices]
    shuffled_y = combined_y[indices]

    return shuffled_x, shuffled_y



def save_data(data: np.ndarray, folder_name: str, file_name: str) -> None:
    '''Save data to a specified directory.
    
    Args:
    - data (np.ndarray): Data to save.
    - folder_name (str): Name of the folder to save data in.
    - file_name (str): Name of the file to save data as.
    '''
    try:
        os.makedirs(folder_name, exist_ok=True)
        np.save(os.path.join(folder_name, file_name), data)
    except Exception as e:
        print(f"Error saving data: {e}")

def perturbation_experiment(trial_number: int, 
                                  input_dimension: int, 
                                  epochs: int = 20, 
                                  batch_size: int = 128, 
                                  use_pseudorehearsal: bool = False,
                                  loss_function: str = "mse",
                                  optimizer: str = "sgd",
                                  verbose: int = 0) -> None:
    '''Run a basic perturbation experiment with various configurations.
    
    Args:
    - trial_number (int): Identifier for the trial.
    - input_dimension (int): Dimension of the input space.
    - epochs (int, optional): Number of training epochs. Default is 20.
    - batch_size (int, optional): Size of training batch. Default is 128.
    - pseudo_rehearse (bool, optional): Whether to use pseudorehearsal. Default is False.
    - loss_function (str, optional): Loss function for training. Default is "mse".
    - verbose (int, optional): Verbosity mode for training. Default is 0 (silent).
    '''
    # Ensuring reproducibility with seed values
    seed_val = hash(f"Trial Number: {trial_number}Input Dimension{input_dimension}") % (2**32)
    #np.random.seed(seed_val)
    rng = np.random.default_rng(seed_val)
    tf.random.set_seed(seed_val)

    # Initialize models based on the given configurations
    models = initialize_all_models(input_dimension, seed_val)

    # Generate training data by repeating a specific point and its corresponding target
    
    point_of_interest = rng.uniform(0, 1, size=(1, input_dimension))
    random_target_value = rng.standard_normal(size=(1, 1))
    
    base_folder = f"results/input_dim_{input_dimension}_{use_pseudorehearsal}_{optimizer}/trial_{trial_number}"
    
    x_train = np.repeat(point_of_interest, NUM_TRAIN_POINTS, axis=0)
    y_train = np.repeat(random_target_value, NUM_TRAIN_POINTS, axis=0)
    
    # Generate random test points
    random_test_points = rng.uniform(0, 1, size=(NUM_TEST_POINTS, input_dimension))
    
    # Containers for storing model outputs before and after training
    outputs_prior = []
    outputs_after = []

    # Container for storing absolute perturbation values (changes in model outputs)
    absolute_perturbations = []

    for i, model in enumerate(models):
        outputs_prior.append(model.predict(random_test_points, verbose=verbose))
        #model.summary()
        model.compile(optimizer='sgd', loss=loss_function)
        
        # Choose between regular training and pseudorehearsal augmented training
        if not use_pseudorehearsal:
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        else:
            augmented_x, augmented_y = pseudorehearsal(input_dim=input_dimension, 
                                                       num_samples=NUM_PSEUDO_SAMPLES, 
                                                       model=model, 
                                                       train_x=x_train, 
                                                       train_y=y_train,
                                                       seed_val=seed_val)
            model.fit(augmented_x, augmented_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
            del augmented_x, augmented_y

        # Predict after training
        outputs_after.append(model.predict(random_test_points,verbose=verbose))
        
        # Calculate the absolute difference between predictions
        absolute_perturbations.append(np.abs(outputs_after[-1] - outputs_prior[-1]))

        # Save results for analysis
        model.save(f"{base_folder}/models/model_{i}")
        save_data(outputs_prior[-1], f"{base_folder}/model_outputs/model_{i}", "output_prior_training.npy")
        save_data(outputs_after[-1], f"{base_folder}/model_outputs/model_{i}", "output_after_training.npy")
        save_data(absolute_perturbations[-1], f"{base_folder}/perturbations/model_{i}", "absolute_perturbation.npy")

    # Compute distances for further analysis
    max_distances = np.linalg.norm((point_of_interest-random_test_points), ord=np.inf, axis=-1)
    min_distances = np.linalg.norm((point_of_interest-random_test_points), ord=-np.inf, axis=-1)
    save_data(min_distances, f"{base_folder}/distances", "min_distances.npy")
    save_data(max_distances, f"{base_folder}/distances", "max_distances.npy")
    
    del models, outputs_prior, outputs_after, absolute_perturbations
    gc.collect()
