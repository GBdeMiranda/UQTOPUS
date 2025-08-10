"""
UQ Runner for OpenFOAM Studies

Main orchestration logic for uncertainty quantification studies.
"""
import os
import subprocess
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pyDOE3 import lhs

from openfoam_tools import load_config

def uq_simulation(X, Params):
    """
    Function to run openFOAM simulations for an experimental design (ED) defined 
    by a table of input parameters. The function creates a directory for each sample (row) in the ED.
    The execution happens for each sample in the ED and the outputs are saved in
    '{experiment_name}'. 

    Parameters:
        X(ndarray)
            N_rv-column matrix with sampled values of the input parameters (N_rv = len(Params['model_parameters']): number of random variables )
            X[:, i] (i=0,1,...,N_in-1): Input parameters

        Params (dict)
            Dictionary containing information about the input and output parameters of the model.
            'experiment': Name of the experiment/folder (default: 'temp')
            'parameter_ranges': Dictionary defining the ranges for each parameter
            'nthreads': Number of threads to be used in the simulation (default: 1)
            'solver': Name of the script-solver to be used. The same as defined in the OpenFOAM template case
    """


    ##############################################################################################################
    ## Input parameters validation ###############################################################################
    for k in Params.keys():
        if k not in [
            'experiment', 'parameter_ranges',
            'nthreads', 'solver',
            'theModel' # parameter from uqpylab, it is not used here
        ]:
            raise Exception(f"Unknown key '{k}' in Params")
    for k in Params['experiment'].keys():
        if k not in ['experiment', 'base_case_dir', 'name']:
            raise Exception(f"Unknown key '{k}' in Params['experiment']")

    base_case_dir = Params['experiment']['base_case_dir'] if 'base_case_dir' in Params['experiment'] else None
    solver = Params['solver'] if 'solver' in Params else None
    keys = list(Params['parameter_ranges'].keys()) if 'parameter_ranges' in Params else None

    if keys is None or solver is None or base_case_dir is None:
        raise Exception("The parameters 'base_case_dir', 'solver', and 'parameter_ranges' must be provided as arguments in Params")
    else:
        if not os.path.exists(base_case_dir):
            raise ValueError('The "base_case_dir" path passed as parameter does not exist')
        if not isinstance(keys, list):
            try:
                keys = list(keys)
            except:
                raise ValueError('The parameter "parameter_ranges" should be a dict with valid inputs')
        if len(keys) != X.shape[1]:
            raise ValueError('The number of keys passed must be equal to the number of the input columns in the experimental design')


    nthreads = Params['nthreads'] if 'nthreads' in Params else 1
    exp_name = Params['experiment']['name'] if 'name' in Params['experiment'] else 'temp'
    ##############################################################################################################

    
    ##############################################################################################################
    ## Sample generation #########################################################################################
    process_func = partial(
        _process_simulation,
        exp_config=Params
    )

    iparams = list(enumerate([ dict(zip(keys, x)) for x in X ]))
    with mp.get_context('spawn').Pool(nthreads) as pool:
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


def _process_simulation(param_data, exp_config, verbose=False):
    """
    Process a single simulation (helper function for multiprocessing).
    
    Parameters:
        param_data ((index, parameters_dict)): Tuple containing the sample index and parameters dictionary.
        base_case_dir (str): Path to base case directory
        exp_name (str): Experiment name
        exp_config (dict): Solver configuration
    """
    i, params = param_data
    exp_name = exp_config['experiment']['name']
    experiment_name = f"{exp_name}/sample_{i:03d}"
    exp_config['experiment']['name'] = experiment_name
    
    try:
        run_simulation(
            params=params,
            exp_config=exp_config,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error in sample {i}: {e}")





def generate_samples(n_samples, param_ranges, method='lhs', seed=None):
    """Generate parameter samples for UQ study."""
    from pyDOE3 import fullfact, pbdesign, bbdesign, ccdesign
    
    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    if method == 'lhs':
        unit_samples = lhs(n_params, samples=n_samples, criterion='centermaximin')
    elif method == 'random':
        unit_samples = np.random.random((n_samples, n_params))
    elif method == 'grid' or method == 'fullfact':
        n_levels = int(np.ceil(n_samples ** (1/n_params)))
        unit_samples = fullfact([n_levels] * n_params) / (n_levels - 1)
        unit_samples = unit_samples[:n_samples]
    elif method == 'plackett_burman':
        unit_samples = (pbdesign(n_params) + 1) / 2
        unit_samples = unit_samples[:n_samples]
    elif method == 'box_behnken':
        unit_samples = (bbdesign(n_params) + 1) / 2
        unit_samples = unit_samples[:n_samples]
    elif method == 'central_composite':
        unit_samples = (ccdesign(n_params) + 1) / 2
        unit_samples = np.clip(unit_samples, 0, 1)
        unit_samples = unit_samples[:n_samples]
    else:
        raise ValueError(f"Unknown sampling method: {method}. Available methods: 'lhs', 'random', 'grid', 'fullfact', 'plackett_burman', 'box_behnken', 'central_composite'")
    
    samples = np.zeros_like(unit_samples)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)
    
    return samples




def run_simulation(params, exp_config, verbose=False):
    """
    Runs an OpenFOAM simulation with the given parameters.

    Parameters:
        params (dict): Dictionary containing the parameters for the simulation.
        exp_config (dict): Configuration dictionary containing experiment details.
    """

    new_dir = "../experiments/" + exp_config['experiment']['name']
    base_dir = exp_config['experiment']['base_case_dir']

    try:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        else:
            if verbose:
                print(" -- The directory already exists. Files will be overwritten. --")
            
        result = subprocess.run(
            ["rsync", "-av", "--delete", f'{base_dir}/', new_dir + '/'],
            check=True,
            capture_output=True,
            text=True
        )
        if verbose:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error copying the files:", e.stderr)

    env = Environment(
        loader=FileSystemLoader(base_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )

    for param_path, value in params.items():
        path_parts = param_path.split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Parameter key '{param_path}' is not in the correct format. Use 'folder/filename/paramname' format.")
        param = path_parts[-1]
        template_path = os.path.join(*path_parts[:-1])

        template = env.get_template(template_path) # Location of the template file with foam parameters
        
        output = template.render({param: value}, undefined=StrictUndefined)

        # Overwrite the template file with the new values
        with open(os.path.join(new_dir, template_path), 'w') as f:
            f.write(output)

    try:
        current_dir = os.getcwd()
        os.chdir(new_dir)
        result = subprocess.run(
            [f"./{exp_config['solver']}"],
            check=True,
            capture_output=True,
            text=True
        )
        if verbose:
            print(result.stdout)
        os.chdir(current_dir)
    except subprocess.CalledProcessError as e:
        print("Error running the script:", e.stderr)


def run_uq_study(config_file, n_samples, verbose=False):
    """
    Standalone function to run UQ study for scalar parameters.
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
        exp_config=config
    )

    iparams = list(enumerate([ dict(zip(keys, x)) for x in X ]))
    with mp.get_context('spawn').Pool(nthreads) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_func, iparams),
            total=len(iparams), 
            desc='Running simulations',
            mininterval=1.0     # Updates at most once per second
        ):
            pass

    if verbose:
        print(f"UQ study completed. Results saved in 'experiments/{config['experiment']['name']}' folder")
    return None