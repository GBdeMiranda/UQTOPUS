"""
UQ Runner for OpenFOAM Studies

Main orchestration logic for uncertainty quantification studies.
Based on the original working implementation with configuration support.
"""
import os
import numpy as np
import pandas as pd     
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from openfoam_tools import run_simulation, load_config, read_openfoam_field

def model(X, Params):
    """
    Function to run openFOAM simulation for an experimental design (ED) defined 
    by a table of input parameters. The function creates a directory for each sample (row) in the ED.
    The execution happens for each sample in the ED and the outputs are saved in
    '{experiment_name}/outputs'. 
    In general, this is a previous step before training an emulator.

    `Training an emulator will be a separate step to provide flexibility.`
    Parameters:
        X(ndarray)
            N_rv-column matrix with sampled values of the input parameters (N_rv = len(Params['model_parameters']): number of random variables )
            X[:,i] (i=0,1,...,N_in-1): Input parameters

        Params (dict)
            Dictionary containing information about the input and output parameters of the model.
            'base_case_dir': Path to the base case directory
            'experiment_name': Path to the experiment folder (default: 'temp')
            'model_parameters': List of input parameters
            'nthreads': Number of threads to be used in the simulation (default: 1)
            'vector_variables': Dictionary of vector variables if used (default: {})
    """


    ##############################################################################################################
    ## Input parameters validation ###############################################################################
    for k in Params.keys():
        if k not in [
            'experiment',
            'parameter_ranges',
            'nthreads',
            'model_parameters',
            'theModel' # parameter from uqpylab, it is not used here
        ]:
            raise Exception(f"Unknown key '{k}' in Params")
    for k in Params['experiment'].keys():
        if k not in ['name', 'solver', 'base_case_dir']:
            raise Exception(f"Unknown key '{k}' in Params['experiment']")

    keys = [k.split('/')[-1] for k in Params['parameter_ranges'].keys()] if 'parameter_ranges' in Params else None

    base_case_dir = Params['experiment']['base_case_dir'] if 'base_case_dir' in Params['experiment'] else None
    exp_name = Params['experiment']['name'] if 'name' in Params['experiment'] else 'temp'
    solver_config = Params['solver_config'] if 'solver_config' in Params else None
    nthreads = Params['nthreads'] if 'nthreads' in Params else 1

    if base_case_dir is None or keys is None or solver_config is None:
        raise Exception("The parameters 'base_case_dir', 'model_parameters', and 'solver_config' must be provided as arguments in Params")
    else:
        if not os.path.exists(base_case_dir):
            raise ValueError('The "base_case_dir" path passed as parameter does not exist')
        if not isinstance(keys, list):
            try:
                keys = list(keys)
            except:
                raise ValueError('The parameter "model_parameters" is not a list or cannot be converted to a list')
        if len(keys) != X.shape[1]:
            raise ValueError('The number of keys must be equal to the number of the input columns in the experimental design')
    ##############################################################################################################

    
    ##############################################################################################################
    ## Sample generation #########################################################################################
    process_func = partial(
        _process_simulation,
        exp_name=exp_name,
        solver_config=solver_config
    )

    iparams = list(enumerate([ dict(zip(keys, x)) for x in X ]))
    with Pool(nthreads) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_func, iparams),
            total=len(iparams), 
            desc='Running simulations',
            mininterval=1.0     # Updates at most once per second
        ):
            pass
    ##############################################################################################################

    print(f"Simulation executed successfully. Files saved in 'experiments/{exp_name}' folder")

    return None


def _process_simulation(param_data, experiment_config):
    """
    Process a single simulation (helper function for multiprocessing).
    
    Parameters:
        param_data ((index, parameters_dict)): Tuple containing the sample index and parameters dictionary.
        base_case_dir (str): Path to base case directory
        exp_name (str): Experiment name
        solver_config (dict): Solver configuration
    """
    i, params = param_data
    exp_name = experiment_config['experiment']['name']
    experiment_name = f"{exp_name}/sample_{i:03d}"
    experiment_config['experiment']['name'] = experiment_name
    # print(experiment_config)
    try:
        run_simulation(
            params=params,
            experiment_config=experiment_config,
            verbose=False
        )
    except Exception as e:
        print(f"Error in sample {i}: {e}")





def generate_samples(n_samples, param_ranges, method='lhs', seed=None):
    """Generate parameter samples for UQ study."""
    # TODO: Modify to use always pydoe2 for sampling and choose all experimental designs

    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    if method == 'lhs':
        try:
            from pyDOE2 import lhs
            unit_samples = lhs(n_params, samples=n_samples, criterion='centermaximin')
        except ImportError:
            print("Warning: pyDOE2 not available, using random sampling")
            unit_samples = np.random.random((n_samples, n_params))
    elif method == 'random':
        unit_samples = np.random.random((n_samples, n_params))
    elif method == 'grid':
        n_levels = int(np.ceil(n_samples ** (1/n_params)))
        axes = [np.linspace(0, 1, n_levels) for _ in range(n_params)]
        grid = np.meshgrid(*axes)
        unit_samples = np.column_stack([g.ravel() for g in grid])
        unit_samples = unit_samples[:n_samples]
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Scale to parameter ranges
    samples = np.zeros_like(unit_samples)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
    
    return samples



def run_uq_study(config_file, n_samples):
    """
    Standalone function to run UQ study (alternative to UQPyLab interface).
    """
    config = load_config(config_file)

    X = generate_samples(
        n_samples=n_samples,
        param_ranges=config['parameter_ranges'],
        method='lhs',
        seed=42
    )
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    X = X.tolist()

    nthreads = config['nthreads'] if 'nthreads' in config else 1
    keys = config['parameter_ranges'].keys()
    if keys is None:
        raise Exception("The parameter 'parameter_ranges' must be provided in the config file")

    process_func = partial(
        _process_simulation,
        experiment_config=config
    )

    iparams = list(enumerate([ dict(zip(keys, x)) for x in X ]))
    with Pool(nthreads) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_func, iparams),
            total=len(iparams), 
            desc='Running simulations',
            mininterval=1.0     # Updates at most once per second
        ):
            pass

    print(f"UQ study completed. Results saved in 'experiments/{config['experiment']['name']}' folder")
    return None