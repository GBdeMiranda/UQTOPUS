"""
Adaptive Simulator (Active Learning / Uncertainty-Aware Surrogate)

A hybrid simulator that chooses dynamically between a fast emulator (surrogate model)
and a high-fidelity OpenFOAM simulator based on prediction uncertainty.
"""

from __future__ import annotations
import logging
from typing import Callable, Any
import xarray as xr

# Internal package imports
from .simulation import OpenFOAMSimulator

logger = logging.getLogger(__name__)


class AdaptiveSimulator:
    """
    Hybrid simulator implementing an Active Learning loop.

    Dynamically routes simulation requests to a fast surrogate model (emulator)
    when prediction uncertainty is low, and falls back to a high-fidelity
    OpenFOAM simulator when the region is unexplored or uncertainty is high.

    Parameters:
        real_simulator (OpenFOAMSimulator):
            The high-fidelity OpenFOAM simulator instance.

        emulator (Any):
            The surrogate model. Must implement a `predict(params)` method
            returning an `xr.Dataset`.

        threshold (float):
            The uncertainty threshold. If calculated uncertainty is strictly
            greater than this value, the real simulator is executed.

        uncertainty_fn (Callable[[dict[str, float]], float] or Callable[[Any], float], optional):
            Custom callback to compute the uncertainty.
            If provided, it will be called as `uncertainty_fn(params)` or `uncertainty_fn(prediction)`.
            If None, the simulator attempts to call `emulator.predict_uncertainty(params)`
            or checks if `emulator.predict(params)` returns a tuple of `(prediction, uncertainty)`.

        update_fn (Callable[[dict[str, float], xr.Dataset], None], optional):
            Optional callback to update/retrain the emulator online when a
            real high-fidelity simulation is executed. Called as `update_fn(params, real_dataset)`.
    """

    def __init__(
        self,
        real_simulator: OpenFOAMSimulator,
        emulator: Any,
        threshold: float,
        uncertainty_fn: Callable[[dict[str, float]], float]
        | Callable[[Any], float]
        | None = None,
        update_fn: Callable[[dict[str, float], xr.Dataset], None] | None = None,
    ) -> None:
        self.real_simulator = real_simulator
        self.emulator = emulator
        self.threshold = threshold
        self.uncertainty_fn = uncertainty_fn
        self.update_fn = update_fn

        self._real_runs: int = 0
        self._emulated_runs: int = 0

    @property
    def real_runs(self) -> int:
        """Total number of high-fidelity OpenFOAM runs executed."""
        return self._real_runs

    @property
    def emulated_runs(self) -> int:
        """Total number of fast surrogate model predictions executed."""
        return self._emulated_runs

    @property
    def run_count(self) -> int:
        """Total number of runs (real + emulated) executed by this instance."""
        return self._real_runs + self._emulated_runs

    def run(
        self,
        params: dict[str, float],
        step: int | None = None,
        verbose: bool = False,
        cleanup: bool = False,
    ) -> xr.Dataset:
        """
        Run a simulation step using either the emulator or the real simulator.

        Parameters:
            params (dict): Parameter values keyed as 'folder__filename__paramname'.
            step (int, optional): Step index passed to the real simulator if invoked.
            verbose (bool): Forward verbose output to the real simulator or logging.
            cleanup (bool): Cleanup temporary directories after real runs.
        """
        uncertainty = self._get_uncertainty(params)

        if uncertainty <= self.threshold:
            if verbose:
                logger.info(
                    "AdaptiveSimulator: Uncertainty (%.4f) <= threshold (%.4f). Using emulator.",
                    uncertainty,
                    self.threshold,
                )
            prediction = self._get_prediction(params)
            self._emulated_runs += 1
            return prediction
        else:
            if verbose:
                logger.info(
                    "AdaptiveSimulator: Uncertainty (%.4f) > threshold (%.4f). Falling back to OpenFOAM.",
                    uncertainty,
                    self.threshold,
                )
            real_result = self.real_simulator.run(
                params, step=step, verbose=verbose, cleanup=cleanup
            )
            self._real_runs += 1

            # Trigger online retraining / model update if callback is set
            if self.update_fn is not None:
                if verbose:
                    logger.info("AdaptiveSimulator: Triggering emulator update/retraining.")
                self.update_fn(params, real_result)

            return real_result

    def reset(self) -> None:
        """Reset internal run counts and the underlying real simulator count."""
        self._real_runs = 0
        self._emulated_runs = 0
        self.real_simulator.reset()

    def _get_uncertainty(self, params: dict[str, float]) -> float:
        """Helper to extract or calculate uncertainty for the given parameters."""
        if self.uncertainty_fn is not None:
            try:
                # Try calling with params
                return float(self.uncertainty_fn(params))  # type: ignore
            except (TypeError, ValueError):
                # Fallback: get prediction first, then pass to uncertainty_fn
                pred = self._get_prediction(params)
                return float(self.uncertainty_fn(pred))  # type: ignore

        if hasattr(self.emulator, "predict_uncertainty"):
            return float(self.emulator.predict_uncertainty(params))

        # Check if predict returns a tuple (prediction, uncertainty)
        pred_res = self.emulator.predict(params)
        if isinstance(pred_res, tuple) and len(pred_res) == 2:
            return float(pred_res[1])

        # Default fallback if no uncertainty method is found
        logger.warning(
            "AdaptiveSimulator: No uncertainty method found on emulator. "
            "Assuming uncertainty = infinity (always falling back to real simulator)."
        )
        return float("inf")

    def _get_prediction(self, params: dict[str, float]) -> xr.Dataset:
        """Helper to get prediction dataset from the emulator."""
        pred_res = self.emulator.predict(params)
        if isinstance(pred_res, tuple) and len(pred_res) == 2:
            return pred_res[0]
        return pred_res

    def __repr__(self) -> str:
        return (
            f"AdaptiveSimulator("
            f"real_runs={self._real_runs}, "
            f"emulated_runs={self._emulated_runs}, "
            f"threshold={self.threshold})"
        )
