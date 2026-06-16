# RL Reward Functions in UQTOPUS

The UQTOPUS RL module (`OpenFOAMEnv`) provides complete flexibility to define reward functions directly on top of OpenFOAM's output data. 

Instead of hardcoding a specific metric (like drag reduction), `OpenFOAMEnv` asks you to provide a `reward_fn` callback. This callback receives a highly structured `xarray.Dataset` containing the full 3D fields (velocity, pressure, etc.) and returns a single float (the reward).

## How it works

When you instantiate `OpenFOAMEnv`, you pass a standard Python function.

```python
import numpy as np
import uqtopus as uqt

def my_custom_reward(dataset):
    # 'dataset' is an xarray.Dataset containing all fields for the last timestep.
    
    # Example 1: Penalize high pressure drop
    # Get pressure field as a numpy array
    p = dataset['p'].values
    pressure_drop = np.max(p) - np.min(p)
    
    # Example 2: Target a specific average velocity in the X direction
    U = dataset['U'].values  # Shape: (time, cell, component)
    avg_velocity_x = np.mean(U[0, :, 0])
    velocity_penalty = abs(avg_velocity_x - 1.5)
    
    # Calculate scalar reward (we maximize reward, so we use negative penalties)
    reward = - (pressure_drop * 0.1) - (velocity_penalty * 1.0)
    return float(reward)

# Pass the function to the environment
env = uqt.OpenFOAMEnv(
    simulator=sim,
    param_ranges={'constant__physicalProperties__nu': (1e-5, 1e-3)},
    observation_fn=my_obs_fn,
    reward_fn=my_custom_reward,
    obs_shape=(50,)
)
```

## Available Outputs

The `dataset` passed to the `reward_fn` contains all variables that were extracted by the `OpenFOAMSimulator`. 

Because `xarray` preserves dimensions and coordinates, you can easily compute complex spatial rewards:
- **Spatial gradients** (e.g., flow uniformity)
- **Local sampling** (e.g., pressure exactly at the wall or at `x = 0.5`)
- **Boundary checks** (e.g., heavily penalize the agent if temperature exceeds a threshold anywhere in the domain)

You can define *any* mathematical function over the raw simulation data to guide your reinforcement learning agent!
