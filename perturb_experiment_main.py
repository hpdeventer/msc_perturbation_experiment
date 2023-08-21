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
    
experiment_arguments = {
    'trial_number': 1,
    'input_dimension': 6,
    'partition_number': 10,
    'use_pseudorehearsal': True,
    'loss_function': 'mae',
    'verbose': 0.,
    'optimizer': 'adam'
}

# Execute the experiment
basic_perturbation_experiment(**experiment_arguments)
