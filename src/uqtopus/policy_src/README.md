# libuqtopusPolicy

One OpenFOAM library, `libuqtopusPolicy.so`. The solver loads it and finds two new types in it: a boundary condition and a source term.

| file | what it is | depends on |
|---|---|---|
| `onnxPolicy.C/.H` | holds the ONNX Runtime session. Observation in, action vector out | ONNX Runtime, basic OpenFOAM types |
| `uqtopusController.C/.H` | one decision per control step, shared by every target. Owns the probes, the schedule, the ramp, the bounds and the trajectory file | `onnxPolicy`, finiteVolume |
| `uqtopusBoundaryConditionFvPatchField.C/.H` | the boundary condition, registered as `uqtopusBoundaryCondition`. Takes one component of the action and writes it on its patch | `uqtopusController` |
| `uqtopusSource.C/.H` | the source term, an `fvModel` registered as `uqtopusSource`. Takes one component of the action and adds it to the equation of one field in a set of cells | `uqtopusController` |

`uqtopusController` lives in the mesh registry, so N targets driven by the same policy see the same action vector.  Any new kind of target finds it the same way, with `uqtopusController::New(mesh, dict)`, and reads its own component.

## What the boundary condition does

Once per control interval:

1. `uqtopusController` reads the field named in the dictionary;
2. it hands them to `onnxPolicy`, which draws one standard normal number per action component, runs the graph and returns the action vector;
3. it ramps linearly from the previous action over `rampFraction` of the interval, and appends one row to `postProcessing/uqtopusPolicy/<startTime>/trajectory.dat`;
4. every target takes its own component and applies it. The boundary condition writes `direction * action[component]` on the faces of its patch.

The graph returns the sampled action, and the controller clips it to the action bounds. The draw is made in the solver, and not inside the graph, because an ONNX random operator takes its seed as a graph attribute: every episode of one training iteration shares a single policy file, so an in-graph draw would make them all identical.

Two guards matter and neither raises an error when it is missing:

- OpenFOAM calls `updateCoeffs()` once per outer corrector of the PIMPLE loop, up to fifteen times per time step in this case. Evaluating there would apply an action different from the one recorded. Guarded by `curTimeIndex_`.
- In a decomposed run every process would draw its own number and apply a different velocity to its slice of the patch. Only the master draws, and the result is scattered.

## The type name says which intervention point

The name in the `type` entry selects a class, and a boundary condition and a source term are different classes. One name per intervention point:

| name | what it drives |
|---|---|
| `uqtopusBoundaryCondition` | a field on a patch |
| `uqtopusSource` | a source term in the equations, an `fvModel` |

`onnxPolicy` and `uqtopusController` are shared by all of them.

## Contract

Implements uqtopus contract.

| | |
|---|---|
| graph input `observation` | float32, shape (1, obs_dim) |
| graph input `noise` | float32, shape (1, act_dim) |
| graph output `action` | float32, shape (1, act_dim), sampled |
| checked from the ONNX metadata | `uqtopus.spec_hash`, `uqtopus.obs_dim`, `uqtopus.act_dim`, `uqtopus.distribution` |

The hash in the dictionary and the hash in the file must agree, otherwise the run stops before the first time step.

Restrictions, all enforced at construction: `stack 1` and a Gaussian distribution.

## Building

### 1. ONNX Runtime

A prebuilt release is enough; there is nothing to compile.

```bash
mkdir -p ~/opt && cd ~/opt
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar xzf onnxruntime-linux-x64-1.17.3.tgz
echo 'export ONNXRUNTIME_ROOT=$HOME/opt/onnxruntime-linux-x64-1.17.3' >> ~/.bashrc
```

### 2. Compile

```bash
source /opt/openfoam9/etc/bashrc         # wherever yours lives
export ONNXRUNTIME_ROOT=$HOME/opt/onnxruntime-linux-x64-1.17.3
cd src/uqtopus/policy_src
wmake libso
```

The result is `$FOAM_USER_LIBBIN/libuqtopusPolicy.so`. This directory can live anywhere on disk; it has no relation to any case.

### 3. Check

```bash
ls $FOAM_USER_LIBBIN/libuqtopusPolicy.so
```

## Using it

In the case, `system/controlDict`:

```
application     pimpleFoam;
libs            ("libuqtopusPolicy.so");
```

The `libs` entry is what puts `uqtopusBoundaryCondition` in the table of boundary condition types. Without it the run stops while reading `0/U`, printing every type OpenFOAM does know.

The contract is a top level entry of the same `controlDict`, named `uqtopusPolicy`, and it is what `uqtopus.rl.render_controller()` produces. OpenFOAM ignores entries it does not know, and the controller reads this one itself:

```
uqtopusPolicy
{
    policy          "/abs/path/policy.onnx";
    specHash        "a1b2c3d4e5f60718";
    controlInterval 0.4;
    startTime       0;
    seed            7;
    observation     { ... }
    action          { ... }
}
```

A patch takes the component of the target that carries its name, and `action` names the target when the two differ. The patch writes `direction` times that component. `direction` is required on a vector or tensor field and defaults to 1 on a scalar field. With `origin`, the patch is instead a wall rotating about `origin` and `axis`, with the component as angular velocity, and `axis` is required:

```
    actuator
    {
        type            uqtopusBoundaryCondition;
        direction       (1 0 0);
        value           uniform (0 0 0);
    }
```

A source term goes in `constant/fvModels`, selects its cells like any `fvModel`, and takes the component of the target that carries its entry name. `direction` follows the same rule as on a patch:

```
heater
{
    type            uqtopusSource;
    field           T;
    selectionMode   cellZone;
    cellZone        heater;
    volumeMode      specific;
}
```