from pathlib import Path
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import xarray as xr
import yaml
from fluidfoam import readmesh, readfield, readvector, readscalar, typefield


def parse_openfoam_case(case_dir:str, variables:list[str], time_dirs:list[str]|str=None):
    """
    Parses the OpenFOAM case directory structure and reads all field data.
    
    Parameters:
        case_dir (str): Path to the root directory of the OpenFOAM case.
        variables (list): List of field names to read.
        time_dirs (list or str, optional): List of time directories to read.
        
    Returns:
        xr.Dataset: Dataset with variables as data variables and time as coordinate.
    """

    if time_dirs is None:
        time_dirs = sorted(
            [p.name for p in Path(case_dir).iterdir() if p.is_dir() and p.stem.isdigit()],
            key=lambda x: float(x)
        )
    else:
        if isinstance(time_dirs, str):
            time_dirs = [time_dirs]

    time_dirs = [str(t) for t in time_dirs]

    # Store all data
    data_vars = {}
    times = [float(t) for t in time_dirs]
    
    # Read all data first
    all_data = {}
    for time_dir in time_dirs:
        all_data[time_dir] = {}
        
        for field_file in variables:
            try:
                all_data[time_dir][field_file] = readfield(case_dir, time_dir, field_file, verbose=False).T
            except Exception as e:
                print(f"Error reading {field_file} in {time_dir}: {e}")
    
    
    x, y, z = readmesh(case_dir, verbose=False)

    # Handling uniform fields (single value in file)
    max_elements = len(x)
    for time_data in all_data.values():
        for fname, field in time_data.items():
            if field.ndim == 1 and field.shape[0] == 1:     # scalar uniform field
                time_data[fname] = np.stack([field] * max_elements, axis=0).flatten()
            elif field.ndim == 2 and (field.shape[0] == 1 or field.shape[1] == 1):   # vector uniform field
                time_data[fname] = np.stack([field[0]] * max_elements, axis=0).reshape(max_elements, -1)

    # Create xarray data variables
    for var in variables:
        # Stack time data for this variable
        var_data = []
        for time_dir in time_dirs:
            var_data.append(all_data[time_dir][var])
        
        if var_data:
            var_array = np.stack(var_data, axis=0)
            
            # Create appropriate dimensions based on shape
            if var_array.ndim == 2: # (time, cell) or (time, cell, component)
                dims = ['time', 'cell']
            elif var_array.ndim == 3:
                dims = ['time', 'cell', 'component']
            else: # higher dimensions
                dims = ['time'] + [f'dim_{i}' for i in range(1, var_array.ndim)]
            
            data_vars[var] = xr.DataArray(var_array, dims=dims)
    
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



def read_uq_experiment(case_dir:str, variables:list[str], n_samples:int, time_dirs:list[str]|str=None, nthreads:int=1):
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

    case_dir = Path(case_dir)

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
                [str(case_dir / f"sample_{i:03d}") for i in range(n_samples)],
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




def load_config(config_path:str="config.yaml"):
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


# ===================================
# LEGACY FUNCTIONS ==================
# Not imported to the main module ===
# ===================================

import re

def read_openfoam_field(file_path):
    """
    LEGACY
    Reads an OpenFOAM field file and returns the data as a NumPy array.

    Parameters:
        file_path (str): Path to the OpenFOAM field file.

    Returns:
        np.ndarray: NumPy array with the field data.
    """

    print("Warning: Legacy function. Use fluidfoam.readfield instead!")

    try:
        with open(file_path, 'r') as f:
            content = f.readlines()
        
        # Find the 'internalField' line
        start_index = next(i for i, line in enumerate(content) if line.startswith('internalField'))

        # Check if the field is uniform
        field_info = content[start_index]
        if field_info.split()[1] == 'uniform':

            data = re.findall(r"[-+]?\d*\.\d+|\d+", field_info)
            values = np.array([float(d) for d in data])
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