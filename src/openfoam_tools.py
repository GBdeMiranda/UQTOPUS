import os
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import re
import yaml
from fluidfoam import readmesh, readvector, readscalar  # XXX Change to fluidfoam reading strategy?


def read_openfoam_field(file_path):
    """
    Reads an OpenFOAM field file and returns the data as a NumPy array.

    Parameters:
        file_path (str): Path to the OpenFOAM field file.

    Returns:
        np.ndarray: NumPy array with the field data.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.readlines()
        
        # Find the 'internalField' line
        start_index = next(i for i, line in enumerate(content) if line.startswith('internalField'))

        # Check if the field is uniform
        field_info = content[start_index]
        if field_info.split()[1] == 'uniform':

            # Considering 1D domain
            data = re.findall(r"[-+]?\d*\.\d+|\d+", field_info)
            values = np.array([float(data[0])])
            return values
            
        # Non uniform has the number of elements in the data block
        num_elements = int(content[start_index + 1])
        
        # Extract the data block
        data = content[start_index + 3:start_index + 3 + num_elements]
        
        # Parse data into NumPy array
        values = []
        for line in data:
            line = line.strip().strip('()')
            if ' ' in line:  # Vector or multiple values
                try:
                    values.append(np.array([float(x) for x in line.split()]))
                except ValueError:
                    print(f"Warning: Skipping malformed vector line: {line}")
            else:  # Single value
                try:
                    values.append(float(line))
                except ValueError:
                    print(f"Warning: Skipping malformed scalar line: {line}")

        return np.array(values)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None



def parse_openfoam_case(case_dir, variables, time_dirs=None):
    """
    Parses the OpenFOAM case directory structure and reads all field data.
    
    Parameters:
        case_dir (str): Path to the root directory of the OpenFOAM case.
        variables (list): List of field names to read.
        time_dirs (list or str, optional): List of time directories to read.
        
    Returns:
        xr.Dataset: Dataset with variables as data variables and time as coordinate.
    """
    import xarray as xr
    
    # Get time directories
    if time_dirs is None:
        time_dirs = sorted([d for d in os.listdir(case_dir) if d.isdigit()], key=lambda x: float(x))
    else:
        if type(time_dirs) == str:
            time_dirs = [time_dirs]
        time_dirs = [str(t) for t in time_dirs]
    
    # Store all data
    data_vars = {}
    times = [float(t) for t in time_dirs]
    
    # Read all data first
    all_data = {}
    for time_dir in time_dirs:
        time_path = os.path.join(case_dir, time_dir)
        all_data[time_dir] = {}
        
        for field_file in variables:
            field_path = os.path.join(time_path, field_file)
            try:
                all_data[time_dir][field_file] = read_openfoam_field(field_path)
            except Exception as e:
                print(f"Error reading {field_file} in {time_dir}: {e}")
    
    # Handle uniform fields
    max_elements = max([len(all_data[t][f]) for t in all_data for f in all_data[t] if f in all_data[t]])
    
    for time_dir in all_data:
        for field in all_data[time_dir]:
            if len(all_data[time_dir][field]) == 1:
                all_data[time_dir][field] = np.repeat(all_data[time_dir][field], max_elements, axis=0)
    
    # Create xarray data variables
    for var in variables:
        # Stack time data for this variable
        var_data = []
        for time_dir in time_dirs:
            if var in all_data[time_dir]:
                var_data.append(all_data[time_dir][var])
        
        if var_data:
            var_array = np.stack(var_data, axis=0)
            
            # Create appropriate dimensions based on shape
            if var_array.ndim == 2:
                dims = ['time', 'cell']
            elif var_array.ndim == 3:
                dims = ['time', 'cell', 'component']
            else:
                dims = ['time'] + [f'dim_{i}' for i in range(1, var_array.ndim)]
            
            data_vars[var] = xr.DataArray(var_array, dims=dims)
    
    
    x, y, z = readmesh(case_dir, verbose=False)
    
    ds = xr.Dataset(
        data_vars, 
        coords={
            'time': times,
            'x': ('cell', x),
            'y': ('cell', y),
            'z': ('cell', z)
        }
    )
    
    return ds




def read_uq_experiment(case_dir, variables, n_samples, time_dirs=None, nthreads=4):
    """
    Parses the OpenFOAM case directory structure and reads all field data.
    
    Parameters:
        case_dir (str): Path to the root directory of the OpenFOAM case.
        variables (list): List of field names to read.
        n_samples (int): Number of samples to read.
        time_dirs (list or str, optional): List of time directories to read.
        nthreads (int): Number of parallel jobs.
        
    Returns:
        xr.Dataset: Dataset with sample, time, and cell dimensions.
    """
    import xarray as xr
    from functools import partial
    from multiprocessing import Pool
    from tqdm import tqdm
    
    datasets = []
    sample_ids = []
    
    with mp.get_context('spawn').Pool(nthreads) as pool:
        results = list(tqdm(
            pool.imap(
                partial(
                    parse_openfoam_case,
                    variables=variables,
                    time_dirs=time_dirs
                ),
                [f'{case_dir}/sample_{i:03d}' for i in range(n_samples)],
            ),
            total=n_samples,
            desc="Processing cases", 
            unit="case",
            mininterval=1.0
        ))
    
    # Collect datasets with sample indices
    for i, ds in enumerate(results):
        datasets.append(ds)
        sample_ids.append(i)
    
    # Concatenate along sample dimension
    combined_ds = xr.concat(datasets, dim='sample')
    combined_ds = combined_ds.assign_coords(sample=sample_ids)
    
    return combined_ds



def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Parameters:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}