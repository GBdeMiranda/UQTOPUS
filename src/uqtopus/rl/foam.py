"""
Controller Dictionary Rendering

Turns a PolicySpec into a standalone OpenFOAM dictionary block holding the probes,
targets, bounds, control interval, ramp, policy file and contract hash.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .spec import PolicySpec

INDENT = "    "

_BARE_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]*$")


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.capitalize() for word in rest)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar(value: Any) -> str:
    """One OpenFOAM token."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if _is_number(value):
        text = repr(float(value)) if isinstance(value, float) else str(value)
        return text.rstrip("0").rstrip(".") if "." in text and "e" not in text else text
    text = str(value)
    return text if _BARE_WORD.match(text) else f'"{text}"'


def _is_number_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(_is_number(v) for v in value)


def format_value(value: Any, depth: int) -> str:
    """
    Render a Python value as the right-hand side of a dictionary entry.

    Numbers, strings and booleans become single tokens; flat number sequences
    become '(a b c)'; nested sequences and sequences of mappings spread over
    several lines; mappings become braced blocks.
    """
    pad = INDENT * depth

    if isinstance(value, Mapping):
        return format_block(value, depth)

    if isinstance(value, (list, tuple)):
        if not value:
            return "()"
        if _is_number_sequence(value):
            return "(" + " ".join(_scalar(v) for v in value) + ")"
        lines = [f"{pad}("]
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{pad}{INDENT}" + format_block(item, depth + 1).lstrip())
            else:
                lines.append(f"{pad}{INDENT}{format_value(item, depth + 1)}")
        lines.append(f"{pad})")
        return "\n".join(lines)

    return _scalar(value)


def format_block(mapping: Mapping[str, Any], depth: int = 0) -> str:
    """
    Render a mapping as a braced OpenFOAM sub-dictionary.

    The terminator depends on the value: a sub-dictionary takes none, a list
    takes a semicolon after the closing parenthesis, and a plain entry takes
    one after the value.
    """
    pad = INDENT * depth
    lines = [f"{pad}{{"]
    for key, value in mapping.items():
        if value is None:
            continue

        if isinstance(value, Mapping):
            lines.append(f"{pad}{INDENT}{key}")
            lines.append(format_block(value, depth + 1))
        elif isinstance(value, (list, tuple)) and value and not _is_number_sequence(value):
            lines.append(f"{pad}{INDENT}{key}")
            lines.append(format_value(value, depth + 1) + ";")
        else:
            rendered = format_value(value, depth + 1)
            lines.append(f"{pad}{INDENT}{key}{' ' * max(1, 16 - len(key))}{rendered};")
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def format_entries(mapping: Mapping[str, Any], depth: int = 0) -> str:
    """Render a mapping as bare entries, without the enclosing braces."""
    block = format_block(mapping, depth)
    return "\n".join(block.splitlines()[1:-1])


# The controller block

def controller_mapping(
    spec: PolicySpec,
    policy: str | Path,
    *,
    controller_type: str = "uqtopusBoundaryCondition",
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the controller entries as a plain mapping, before rendering.

    Exposed separately so a case that needs a different layout can reshape the
    mapping instead of parsing the rendered text.
    """
    observation = {
        "dim": spec.obs_dim,
        "sources": [
            {_camel(k): v for k, v in source.to_dict().items() if v is not None}
            for source in spec.observation.sources
        ],
    }

    action = {
        "name": spec.action.name,
        "nComponents": spec.action.n_components,
        "distribution": spec.action.distribution,
        "rampFraction": spec.action.ramp_fraction,
        "low": list(spec.action.low),
        "high": list(spec.action.high),
        "targets": [
            {"name": name, "component": index}
            for index, name in enumerate(spec.action.targets)
        ],
    }

    mapping: dict[str, Any] = {
        "type": controller_type,
        "policy": str(policy),
        "specHash": spec.hash,
        "controlInterval": spec.control_interval,
        "startTime": spec.start_time,
        "endTime": spec.end_time,
        "seed": seed,
        "observation": observation,
        "action": action,
    }
    if extra:
        mapping.update(extra)
    return mapping


def render_controller(
    spec: PolicySpec,
    policy: str | Path,
    *,
    name: str | None = None,
    controller_type: str = "uqtopusBoundaryCondition",
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """
    Render the controller as OpenFOAM dictionary text.

    Parameters:
        spec (PolicySpec): the contract. Its hash is written into the block, so
            a case dictionary that drifts from the policy is caught at startup.
        policy (str or Path): path to the .onnx file, as the solver will see it.
        name (str or None): wrap the entries in a named sub-dictionary. None
            emits bare entries, which is what a boundaryField entry wants.
        controller_type (str): the 'type' entry the solver dispatches on.
            Defaults to the ONNX policy; an MPC controller reusing the same
            observation and action plumbing would pass its own.
        seed (int or None): RNG seed for action sampling, recorded so an
            episode can be replayed.
        extra (mapping or None): additional entries, merged in last.

    Returns:
        str: the dictionary text, without a trailing newline.
    """
    mapping = controller_mapping(
        spec, policy, controller_type=controller_type, seed=seed, extra=extra
    )
    if name is None:
        return format_entries(mapping)
    return f"{name}\n" + format_block(mapping)


def controller_params(
    spec: PolicySpec,
    policy: str | Path,
    keys: str | Sequence[str],
    **kwargs: Any,
) -> dict[str, str]:
    """
    Build the parameter mapping consumed by uqtopus.run_simulation.

    Parameters:
        keys (str or sequence of str): template keys in the package's
            'folder__file__variable' form, e.g. '0__U__controller' to fill a
            {{ controller }} placeholder in the case's 0/U file. Several keys
            render the same block into several files.
    """
    block = render_controller(spec, policy, **kwargs)
    if isinstance(keys, str):
        keys = [keys]
    return {key: block for key in keys}
