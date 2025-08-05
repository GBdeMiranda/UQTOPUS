import subprocess
import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import numpy as np
import re
import pandas as pd
import yaml

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
        **Note: The default list of variables is too expensive for large samples.**

    Parameters:
        case_dir (str): Path to the root directory of the OpenFOAM case.
        variables (list): List of field names to read.
        time_dirs (list or str, optional): List of time directories to read. If None, reads all time directories.
        
    Returns:
        pd.DataFrame: Pandas DataFrame with the field data, where each column is a variable 
            and each row is a time step. Each cell contains an array with the field data.
    """
    data = {}

    # Iterate over time directories, e.g. '50', '100', '200', ...
    if time_dirs is None:
        time_dirs = sorted([d for d in os.listdir(case_dir) if d.isdigit()], key=lambda x: int(x))
    else:
        if type(time_dirs) == str:
            time_dirs = [time_dirs]
        time_dirs = [str(t) for t in time_dirs]

    for time_dir in time_dirs:
        
        time_path = os.path.join(case_dir, time_dir)

        data[time_dir] = {}

        # Iterate over field/var files in the time directory, e.g. 'U', 'p', 'S', ...
        for field_file in variables:
            field_path = os.path.join(time_path, field_file)
            try:
                data[time_dir][field_file] = read_openfoam_field(field_path)
                # print(f"Read {field_file} from {time_dir}")
            except Exception as e:
                print(f"Error reading {field_file} in {time_dir}: {e}")

    # Find the maximum number of elements in the data
    max_elements = max([len(data[time_dir][field]) for time_dir in data for field in data[time_dir]])
    
    for time_dir in data:
        for field in data[time_dir]:
            if len(data[time_dir][field]) == 1:
                data[time_dir][field] = np.repeat(data[time_dir][field], max_elements)

    # Convert to DataFrame
    data = pd.DataFrame(data)
    data = data.transpose()
    data.index = data.index.astype(int)
    
    return data



def run_simulation(params, experiment_config, verbose=True):
    """
    Runs an OpenFOAM simulation with the given parameters.

    Parameters:
        params (dict): Dictionary containing the parameters for the simulation.
        experiment_config (dict): Configuration dictionary containing experiment details.
    """

    new_dir = "../experiments/" + experiment_config['experiment']['name']
    base_dir = experiment_config['experiment']['base_case_dir']

    try:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        else:
            if verbose:
                print(" -- The directory already exists. Files will be overwritten. --")
            
        result = subprocess.run(
            ["cp", "-a", f'{base_dir}/.', new_dir],
            check=True,
            capture_output=True,
            text=True
        )
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
            [f"./{experiment_config['solver']}"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        os.chdir(current_dir)
    except subprocess.CalledProcessError as e:
        print("Error running the script:", e.stderr)



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