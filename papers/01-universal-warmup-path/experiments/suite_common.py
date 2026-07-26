"""Shared, publication-facing utilities for the Paper 1 experiment suite.

The module deliberately contains no machine-specific paths.  Run it from an
environment in which the pinned BlackJAX and tuningfork checkouts are
importable; every runner records and validates their Git revisions.
"""

from __future__ import annotations

import hashlib
import dataclasses
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Release runs are CPU/x64 by protocol.  Set these before importing JAX,
# NumPyro, or ArviZ/Matplotlib rather than inheriting a conflicting shell
# default.
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper1-matplotlib")

import arviz as az
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.adaptation.meta import (
    MultiChainMetaAdaptationCoreState,
    extract_multi_chain_verdict,
)
from blackjax.adaptation.meta._calibration import (
    _ASSUMED_AVG_LEAPFROGS_PER_STEP,
    _DETECTION_BRANCH_BETWEEN_MEANS,
    _DETECTION_BRANCH_BOTH,
    _DETECTION_BRANCH_NONE,
    _DETECTION_BRANCH_POOLED_WITHIN,
    _R_MIN,
)
from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule
from blackjax.adaptation.metric_recipes import REGISTRY
from blackjax.adaptation.staged_adaptation import staged_adaptation
from tuningfork.model import MODELS
from tuningfork.model._numpyro import build_logdensity_fn
from window_events import build_metric_window_events, schedule_prescriptions


CURRENT_BLACKJAX_SHA = "29d2468857be4de1644ca4470c2a4aa7f8137656"
TUNINGFORK_SHA = "79ffb73250f5024dc511b3035d373d11474c2195"
HISTORICAL_SEQUENTIAL_SHA = "2f62921848a93e7dc544ba9de8e29ef177e373b6"
SCHEMA_VERSION = "paper1-experiments-v1"

MC_SCALAR_NAMES = (
    "has_escalated",
    "escalation_rank",
    "r2_latest",
    "r2_mode",
    "budget_used",
    "converged_at_step",
    "airm_vel_prev",
    "airm_vel_curr",
    "is_slow_mixing",
    "chain_collinearity",
    "unimodality_passed",
    "deferred_to_ensemble",
    "within_lam1",
    "chain_consistency_psi",
    "r1_top",
    "detection_branch",
    "unimodality_flag_count",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def canonical_json(row: Mapping[str, Any]) -> str:
    """Encode one deterministic JSONL row."""

    return json.dumps(
        dict(row),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ImmutableJsonl:
    """Append-only JSONL writer that refuses an existing target."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")

    def emit(self, row: Mapping[str, Any]) -> None:
        self._handle.write(canonical_json(row) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ImmutableJsonl":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    """Hash an array together with its dtype and shape."""

    try:
        array = np.ascontiguousarray(np.asarray(value))
    except TypeError:
        # JAX typed PRNG keys intentionally disallow direct NumPy conversion.
        array = np.ascontiguousarray(np.asarray(jax.random.key_data(value)))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def persist_array_artifact(
    path: Path,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable compressed array sidecar and return its manifest."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        name: np.asarray(value)
        for name, value in sorted(arrays.items())
    }
    with artifact_path.open("xb") as handle:
        np.savez_compressed(handle, **normalized)
    return {
        "path": artifact_path.name,
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
        "members": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for name, value in normalized.items()
        },
    }


def _git_root(path: Path) -> Path | None:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def git_state_for_module(module: Any) -> dict[str, Any]:
    """Return revision/dirty state without recording an absolute local path."""

    root = _git_root(Path(module.__file__))
    if root is None:
        return {
            "revision": None,
            "dirty": None,
            "source": "installed-package-no-git-metadata",
        }
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    return {
        "revision": revision,
        "dirty": dirty,
        "source": root.name,
    }


def require_revision(
    module: Any,
    expected: str,
    *,
    label: str,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    state = git_state_for_module(module)
    if state["revision"] != expected:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, "
            f"observed {state['revision']!r}. Install/import the pinned checkout."
        )
    if state["dirty"] and not allow_dirty:
        raise RuntimeError(f"{label} checkout is dirty; refusing a release run.")
    return state


def provenance(
    *,
    suite: str,
    argv: Sequence[str],
    expected_blackjax_sha: str = CURRENT_BLACKJAX_SHA,
    allow_dirty: bool = False,
    run_id: str,
) -> dict[str, Any]:
    import tuningfork

    bjx_state = require_revision(
        blackjax,
        expected_blackjax_sha,
        label="BlackJAX",
        allow_dirty=allow_dirty,
    )
    tf_state = require_revision(
        tuningfork,
        TUNINGFORK_SHA,
        label="tuningfork",
        allow_dirty=allow_dirty,
    )
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"Paper 1 release runs require the CPU backend; got "
            f"{jax.default_backend()!r}."
        )
    if not jax.config.x64_enabled:
        raise RuntimeError("Paper 1 release runs require JAX x64.")
    devices = [
        {
            "platform": device.platform,
            "kind": device.device_kind,
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    portable_argv = [
        Path(value).name if Path(value).is_absolute() else value for value in argv
    ]
    helper_path = Path(__file__).resolve()
    runner_path = Path(argv[0]).resolve()
    source_paths = {helper_path.name: helper_path}
    if runner_path.is_file():
        source_paths[runner_path.name] = runner_path
    return {
        "record_type": "provenance",
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [Path(portable_argv[0]).name, *portable_argv[1:]],
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "os": platform.system(),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpyro": importlib.metadata.version("numpyro"),
        "scipy": importlib.metadata.version("scipy"),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "devices": devices,
        "blackjax": {
            "version": getattr(blackjax, "__version__", "unknown"),
            **bjx_state,
        },
        "tuningfork": tf_state,
        "numpy": np.__version__,
        "arviz": az.__version__,
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_ENABLE_X64",
                "JAX_PLATFORM_NAME",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
            )
        },
    }


def finite_or_none(value: Any) -> float | None:
    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _flatten_position(position: Any) -> np.ndarray:
    return np.asarray(fu.ravel_pytree(position)[0])


def build_multi_chain_cell(
    model: str,
    seed: int,
    *,
    n_chains: int = 8,
    clone_radius: float | None = None,
) -> dict[str, Any]:
    """Build fixed-data, independently initialized chain witnesses."""

    entry = MODELS[model]
    offset = 9500 if clone_radius is not None else 9000
    keys = jax.random.split(jax.random.key(offset + seed), n_chains + 3)
    initial, logdensity_tree, _ = build_logdensity_fn(keys[0], entry)
    _, unravel = fu.ravel_pytree(initial)

    if clone_radius is None:
        inits = jnp.stack(
            [
                fu.ravel_pytree(build_logdensity_fn(keys[1 + idx], entry)[0])[0]
                for idx in range(n_chains)
            ]
        )
        init_design = "init_to_uniform"
    else:
        base = fu.ravel_pytree(initial)[0]
        inits = base[None, :] + clone_radius * jax.random.normal(
            keys[1], (n_chains, base.shape[0])
        )
        init_design = f"tight_clone_r={clone_radius:g}"

    def logdensity_flat(position: jax.Array) -> jax.Array:
        return logdensity_tree(unravel(position))

    return {
        "model": model,
        "seed": seed,
        "d": int(inits.shape[1]),
        "n_chains": n_chains,
        "logdensity_fn": logdensity_flat,
        "inits": inits,
        "warmup_key": keys[-2],
        "sample_key": keys[-1],
        "initialization_design": init_design,
    }


def build_isotropic_cell(
    seed: int,
    *,
    d: int = 20,
    n_chains: int = 8,
    radius: float = 2.0,
) -> dict[str, Any]:
    keys = jax.random.split(jax.random.key(2000 + seed), 3)
    inits = radius * jax.random.normal(keys[0], (n_chains, d))

    def logdensity(position: jax.Array) -> jax.Array:
        return -0.5 * jnp.sum(position**2)

    return {
        "model": f"isotropic_gaussian_d{d}",
        "seed": seed,
        "d": d,
        "n_chains": n_chains,
        "logdensity_fn": logdensity,
        "inits": inits,
        "warmup_key": keys[1],
        "sample_key": keys[2],
        "initialization_design": f"normal_radius={radius:g}",
    }


def dispersion_metrics(inits: Any) -> dict[str, float]:
    values = np.asarray(inits)
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return {
        "between_std_top_pc": float(
            singular_values[0] / np.sqrt(max(values.shape[0] - 1, 1))
        ),
        "between_std_rms": float(np.sqrt(np.mean(centered**2))),
    }


def multi_chain_info_fn(state: Any, info: Any, adaptation_state: Any) -> tuple:
    core = adaptation_state.imm_state
    scalars = tuple(getattr(core, name) for name in MC_SCALAR_NAMES)
    return (
        info.num_integration_steps,
        info.is_divergent,
        info.acceptance_rate,
        adaptation_state.step_size,
        scalars,
        core.mu_star,
    )


def reconstruct_multi_chain_state(
    inverse_mass_matrix: Any,
    scalar_trace: Sequence[Any],
    mu_star: Any,
    *,
    n_chains: int,
    d: int,
) -> MultiChainMetaAdaptationCoreState:
    last = {
        name: np.asarray(values[-1])
        for name, values in zip(MC_SCALAR_NAMES, scalar_trace)
    }
    dummy = jnp.zeros((n_chains, 1, d))
    max_rank = int(np.asarray(inverse_mass_matrix.lam).shape[0])
    return MultiChainMetaAdaptationCoreState(
        inverse_mass_matrix=inverse_mass_matrix,
        mu_star=jnp.asarray(mu_star),
        draws_buffer=dummy,
        grads_buffer=dummy,
        buffer_idx=jnp.int32(0),
        background_split=jnp.int32(0),
        recompute_counter=jnp.int32(0),
        has_escalated=jnp.asarray(last["has_escalated"]),
        escalation_rank=jnp.asarray(last["escalation_rank"]),
        s_gap_prev=jnp.asarray(np.nan),
        s_gap_curr=jnp.asarray(np.nan),
        r2_latest=jnp.asarray(last["r2_latest"]),
        r2_mode=jnp.asarray(last["r2_mode"]),
        budget_used=jnp.asarray(last["budget_used"]),
        converged_at_step=jnp.asarray(last["converged_at_step"]),
        prev_lam=jnp.zeros(max_rank),
        airm_vel_prev=jnp.asarray(last["airm_vel_prev"]),
        airm_vel_curr=jnp.asarray(last["airm_vel_curr"]),
        is_slow_mixing=jnp.asarray(last["is_slow_mixing"]),
        chain_collinearity=jnp.asarray(last["chain_collinearity"]),
        unimodality_passed=jnp.asarray(last["unimodality_passed"]),
        deferred_to_ensemble=jnp.asarray(last["deferred_to_ensemble"]),
        within_lam1=jnp.asarray(last["within_lam1"]),
        chain_consistency_psi=jnp.asarray(last["chain_consistency_psi"]),
        r1_top=jnp.asarray(last["r1_top"]),
        detection_branch=jnp.asarray(last["detection_branch"]),
        unimodality_flag_count=jnp.asarray(last["unimodality_flag_count"]),
    )


def _sample_transform(state: Any, info: Any) -> tuple:
    return state.position, info.num_integration_steps, info.is_divergent


def sample_single_chain(
    algorithm: Any,
    logdensity_fn: Callable,
    parameters: Mapping[str, Any],
    initial_position: Any,
    rng_key: Any,
    *,
    num_draws: int,
    extra_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(parameters)
    kwargs.update(extra_parameters or {})
    sampler = algorithm(logdensity_fn, **kwargs)
    _, (positions, integration_steps, divergences) = (
        blackjax.util.run_inference_algorithm(
            rng_key,
            sampler,
            num_steps=num_draws,
            initial_position=initial_position,
            transform=_sample_transform,
        )
    )
    draws = np.asarray(positions)
    integration_steps_array = np.asarray(integration_steps)
    divergences_array = np.asarray(divergences)
    bulk_ess = np.asarray(az.ess(draws[None, ...], method="bulk"))
    tail_ess = np.asarray(
        az.ess(draws[None, ...], method="tail", prob=(0.05, 0.95))
    )
    return {
        "draws": draws,
        "draws_sha256": sha256_array(draws),
        "bulk_ess_per_dimension": bulk_ess.tolist(),
        "tail_ess_per_dimension": tail_ess.tolist(),
        "min_bulk_ess": float(np.min(bulk_ess)),
        "min_bulk_ess_dimension": int(np.argmin(bulk_ess)),
        "min_tail_ess": float(np.min(tail_ess)),
        "min_tail_ess_dimension": int(np.argmin(tail_ess)),
        "sampling_integration_steps": integration_steps_array,
        "sampling_divergence_trace": divergences_array,
        "sampling_grads": int(integration_steps_array.sum()),
        "sampling_divergences": int(divergences_array.sum()),
    }


def sample_chain_population(
    algorithm: Any,
    logdensity_fn: Callable,
    parameters: Mapping[str, Any],
    initial_positions: Any,
    rng_key: Any,
    *,
    num_draws: int,
    extra_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample every warmup-end witness and report marginal and pooled costs."""

    positions = np.asarray(initial_positions)
    if positions.ndim < 2:
        raise ValueError("population sampling requires a leading chain axis")
    draws: list[np.ndarray] = []
    draws_sha256_per_chain: list[str] = []
    bulk_ess_per_dimension_per_chain: list[list[float]] = []
    tail_ess_per_dimension_per_chain: list[list[float]] = []
    min_bulk_ess_per_chain: list[float] = []
    min_bulk_ess_dimension_per_chain: list[int] = []
    min_tail_ess_per_chain: list[float] = []
    min_tail_ess_dimension_per_chain: list[int] = []
    sampling_integration_steps: list[np.ndarray] = []
    sampling_divergence_trace: list[np.ndarray] = []
    sampling_grads_per_chain: list[int] = []
    sampling_divergences_per_chain: list[int] = []
    for chain_index, initial_position in enumerate(positions):
        key = (
            rng_key
            if chain_index == 0
            else jax.random.fold_in(rng_key, chain_index)
        )
        sampled = sample_single_chain(
            algorithm,
            logdensity_fn,
            parameters,
            initial_position,
            key,
            num_draws=num_draws,
            extra_parameters=extra_parameters,
        )
        draws.append(sampled["draws"])
        draws_sha256_per_chain.append(sampled["draws_sha256"])
        bulk_ess_per_dimension_per_chain.append(
            sampled["bulk_ess_per_dimension"]
        )
        tail_ess_per_dimension_per_chain.append(
            sampled["tail_ess_per_dimension"]
        )
        min_bulk_ess_per_chain.append(sampled["min_bulk_ess"])
        min_bulk_ess_dimension_per_chain.append(
            sampled["min_bulk_ess_dimension"]
        )
        min_tail_ess_per_chain.append(sampled["min_tail_ess"])
        min_tail_ess_dimension_per_chain.append(
            sampled["min_tail_ess_dimension"]
        )
        sampling_integration_steps.append(
            sampled["sampling_integration_steps"]
        )
        sampling_divergence_trace.append(
            sampled["sampling_divergence_trace"]
        )
        sampling_grads_per_chain.append(sampled["sampling_grads"])
        sampling_divergences_per_chain.append(
            sampled["sampling_divergences"]
        )

    pooled_draws = np.stack(draws)
    pooled_bulk_ess = np.asarray(az.ess(pooled_draws, method="bulk"))
    pooled_tail_ess = np.asarray(
        az.ess(pooled_draws, method="tail", prob=(0.05, 0.95))
    )
    pooled_split_rhat = np.asarray(az.rhat(pooled_draws, method="rank"))
    rhat_is_finite = bool(np.isfinite(pooled_split_rhat).all())
    return {
        "draws": pooled_draws,
        "sampling_integration_steps": np.stack(sampling_integration_steps),
        "sampling_divergence_trace": np.stack(sampling_divergence_trace),
        "min_bulk_ess_per_chain": min_bulk_ess_per_chain,
        "min_bulk_ess_dimension_per_chain": (
            min_bulk_ess_dimension_per_chain
        ),
        "min_tail_ess_per_chain": min_tail_ess_per_chain,
        "min_tail_ess_dimension_per_chain": (
            min_tail_ess_dimension_per_chain
        ),
        "bulk_ess_per_dimension_per_chain": (
            bulk_ess_per_dimension_per_chain
        ),
        "tail_ess_per_dimension_per_chain": (
            tail_ess_per_dimension_per_chain
        ),
        "sampling_grads_per_chain": sampling_grads_per_chain,
        "sampling_divergences_per_chain": sampling_divergences_per_chain,
        "draws_sha256_per_chain": draws_sha256_per_chain,
        "min_bulk_ess_pooled": float(np.min(pooled_bulk_ess)),
        "min_bulk_ess_dimension_pooled": int(np.argmin(pooled_bulk_ess)),
        "bulk_ess_per_dimension_pooled": pooled_bulk_ess.tolist(),
        "min_tail_ess_pooled": float(np.min(pooled_tail_ess)),
        "min_tail_ess_dimension_pooled": int(np.argmin(pooled_tail_ess)),
        "tail_ess_per_dimension_pooled": pooled_tail_ess.tolist(),
        "max_split_rhat_pooled": (
            float(np.max(pooled_split_rhat)) if rhat_is_finite else None
        ),
        "max_split_rhat_dimension_pooled": (
            int(np.argmax(pooled_split_rhat)) if rhat_is_finite else None
        ),
        "split_rhat_per_dimension_pooled": [
            finite_or_none(value) for value in pooled_split_rhat
        ],
        "split_rhat_all_finite": rhat_is_finite,
    }


def run_auto_multi_chain(
    cell: Mapping[str, Any],
    *,
    max_grad_budget: int,
    num_draws: int,
    algorithm: Any = blackjax.nuts,
    algorithm_name: str = "nuts",
    extra_parameters: Mapping[str, Any] | None = None,
    capture_events: bool = True,
    sample_all_chains: bool = False,
    draws_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run automatic multi-chain warmup and the declared sampling deliverable."""

    n_chains = int(cell["n_chains"])
    extra_parameters = dict(extra_parameters or {})
    grads_per_step = int(
        extra_parameters.get(
            "num_integration_steps", _ASSUMED_AVG_LEAPFROGS_PER_STEP
        )
    )
    num_warmup_steps = max(
        max_grad_budget // (grads_per_step * n_chains),
        1,
    )
    warmup = staged_adaptation(
        algorithm,
        cell["logdensity_fn"],
        metric="auto",
        max_grad_budget=max_grad_budget,
        n_chains=n_chains,
        adaptation_info_fn=multi_chain_info_fn,
        **extra_parameters,
    )
    start_time = time.monotonic()
    result, trace = warmup.run(
        cell["warmup_key"],
        cell["inits"],
        num_warmup_steps,
    )
    warmup_wall_seconds = time.monotonic() - start_time
    (
        integration_steps,
        warmup_divergence_trace,
        acceptance,
        step_size,
        scalar_trace,
        mu_star,
    ) = trace
    warmup_grads = int(np.asarray(integration_steps).sum())
    warmup_gradient_array = np.asarray(integration_steps)
    if warmup_gradient_array.ndim < 2:
        raise RuntimeError(
            "multi-chain warmup trace lacks the expected leading chain axis"
        )
    warmup_grads_per_chain = (
        warmup_gradient_array.reshape(warmup_gradient_array.shape[0], -1)
        .sum(axis=0)
        .astype(int)
        .tolist()
    )
    warmup_divergences = int(np.asarray(warmup_divergence_trace).sum())
    sample_started = time.monotonic()

    final_core = reconstruct_multi_chain_state(
        result.parameters["inverse_mass_matrix"],
        scalar_trace,
        mu_star[-1],
        n_chains=n_chains,
        d=int(cell["d"]),
    )
    verdict = extract_multi_chain_verdict(
        final_core,
        max_grad_budget,
        num_warmup_steps,
    )
    # Paper convention: sampling starts from the actual warmup-end state.
    terminal_positions = np.asarray(result.state.position)
    if sample_all_chains:
        population_sample = sample_chain_population(
            algorithm,
            cell["logdensity_fn"],
            result.parameters,
            terminal_positions,
            cell["sample_key"],
            num_draws=num_draws,
            extra_parameters=extra_parameters,
        )
        sampled = {
            "min_bulk_ess": population_sample["min_bulk_ess_per_chain"][0],
            "min_bulk_ess_dimension": population_sample[
                "min_bulk_ess_dimension_per_chain"
            ][0],
            "bulk_ess_per_dimension": population_sample[
                "bulk_ess_per_dimension_per_chain"
            ][0],
            "min_tail_ess": population_sample["min_tail_ess_per_chain"][0],
            "min_tail_ess_dimension": population_sample[
                "min_tail_ess_dimension_per_chain"
            ][0],
            "tail_ess_per_dimension": population_sample[
                "tail_ess_per_dimension_per_chain"
            ][0],
            "sampling_grads": population_sample["sampling_grads_per_chain"][0],
            "sampling_divergences": population_sample[
                "sampling_divergences_per_chain"
            ][0],
        }
    else:
        sampled = sample_single_chain(
            algorithm,
            cell["logdensity_fn"],
            result.parameters,
            terminal_positions[0],
            cell["sample_key"],
            num_draws=num_draws,
            extra_parameters=extra_parameters,
        )
        population_sample = None
    sampling_wall_seconds = time.monotonic() - sample_started
    if population_sample is not None:
        sample_arrays = {
            "draws": population_sample["draws"],
            "sampling_integration_steps": population_sample[
                "sampling_integration_steps"
            ],
            "sampling_divergences": population_sample[
                "sampling_divergence_trace"
            ],
        }
    else:
        sample_arrays = {
            "draws": sampled["draws"],
            "sampling_integration_steps": sampled[
                "sampling_integration_steps"
            ],
            "sampling_divergences": sampled[
                "sampling_divergence_trace"
            ],
        }
    array_artifact = (
        persist_array_artifact(
            draws_path,
            {
                **sample_arrays,
                "warmup_integration_steps": warmup_gradient_array,
                "warmup_divergences": np.asarray(
                    warmup_divergence_trace
                ),
                "initial_positions": np.asarray(cell["inits"]),
                "warmup_end_positions": terminal_positions,
            },
        )
        if draws_path is not None
        else None
    )
    flags = verdict.flags
    lam = np.asarray(result.parameters["inverse_mass_matrix"].lam)
    prescriptions = schedule_prescriptions(
        num_warmup_steps,
        n_chains=n_chains,
        dimension=int(cell["d"]),
    )
    sc1_warmup_charge = warmup_grads / n_chains
    sc1_total_grads = sc1_warmup_charge + sampled["sampling_grads"]
    all_chain_warmup_total = warmup_grads + sampled["sampling_grads"]
    row = {
        "record_type": "cell",
        "schema_version": SCHEMA_VERSION,
        "model": cell["model"],
        "seed": int(cell["seed"]),
        "dimension": int(cell["d"]),
        "n_chains": n_chains,
        "comparison_policy": "automatic_joint_population",
        "algorithm": algorithm_name,
        "initialization_design": cell["initialization_design"],
        "initial_positions_sha256": sha256_array(cell["inits"]),
        "warmup_key_sha256": sha256_array(cell["warmup_key"]),
        "sample_key_sha256": sha256_array(cell["sample_key"]),
        "max_grad_budget": max_grad_budget,
        "nominal_max_grad_budget": max_grad_budget,
        "budget_enforcement": "schedule_calibration_not_realized_gradient_stop",
        "warmup_step_conversion_assumed_integration_steps": grads_per_step,
        "num_warmup_steps_per_chain": num_warmup_steps,
        "num_sampling_draws": num_draws,
        "schedule_prescriptions": prescriptions,
        "route": verdict.route,
        "effective_rank": int(verdict.effective_rank),
        "nominal_rank": int(flags["nominal_rank"]),
        "deployed_lam": [float(value) for value in lam],
        "metric_route_status": flags["metric_route_status"],
        "metric_route_basis": flags["metric_route_basis"],
        "metric_scope": flags["metric_scope"],
        "observed_ensemble_evidence": flags["observed_ensemble_evidence"],
        "global_exploration": flags["global_exploration"],
        "handoff": flags["handoff"],
        "confidence_scope": flags["confidence_scope"],
        "chain_collinearity": finite_or_none(flags["chain_collinearity"]),
        "within_lam1": finite_or_none(flags["within_lam1"]),
        "chain_consistency_psi": finite_or_none(flags["chain_consistency_psi"]),
        "r1_top": finite_or_none(flags["r1_top"]),
        "r2_final": finite_or_none(verdict.r2_final),
        "detection_branch": flags["detection_branch"],
        "unimodality_gate": flags["unimodality_gate"],
        "warmup_grads_all_chains": warmup_grads,
        "warmup_grads_per_chain": warmup_grads_per_chain,
        "warmup_grads_sc1_charge": sc1_warmup_charge,
        "warmup_divergences": warmup_divergences,
        "sampling_grads": sampled["sampling_grads"],
        "sampling_divergences": sampled["sampling_divergences"],
        "min_bulk_ess": sampled["min_bulk_ess"],
        "min_bulk_ess_dimension": sampled["min_bulk_ess_dimension"],
        "bulk_ess_per_dimension": sampled["bulk_ess_per_dimension"],
        "min_tail_ess": sampled["min_tail_ess"],
        "min_tail_ess_dimension": sampled["min_tail_ess_dimension"],
        "tail_ess_per_dimension": sampled["tail_ess_per_dimension"],
        "ess_per_grad_sc1": sampled["min_bulk_ess"] / sc1_total_grads,
        "ess_per_grad_marginal_amortized": (
            sampled["min_bulk_ess"] / sc1_total_grads
        ),
        "ess_per_grad_all_warmup_charged": (
            sampled["min_bulk_ess"] / all_chain_warmup_total
        ),
        "ess_per_grad_one_output_total": (
            sampled["min_bulk_ess"] / all_chain_warmup_total
        ),
        "population_sampling_performed": sample_all_chains,
        "warmup_end_positions_sha256": sha256_array(terminal_positions),
        "min_bulk_ess_pooled": (
            population_sample["min_bulk_ess_pooled"]
            if population_sample is not None
            else None
        ),
        "min_bulk_ess_dimension_pooled": (
            population_sample["min_bulk_ess_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "min_tail_ess_pooled": (
            population_sample["min_tail_ess_pooled"]
            if population_sample is not None
            else None
        ),
        "min_tail_ess_dimension_pooled": (
            population_sample["min_tail_ess_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "max_split_rhat_pooled": (
            population_sample["max_split_rhat_pooled"]
            if population_sample is not None
            else None
        ),
        "max_split_rhat_dimension_pooled": (
            population_sample["max_split_rhat_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "split_rhat_all_finite": (
            population_sample["split_rhat_all_finite"]
            if population_sample is not None
            else None
        ),
        "min_bulk_ess_per_chain": (
            population_sample["min_bulk_ess_per_chain"]
            if population_sample is not None
            else None
        ),
        "min_bulk_ess_dimension_per_chain": (
            population_sample["min_bulk_ess_dimension_per_chain"]
            if population_sample is not None
            else None
        ),
        "min_tail_ess_per_chain": (
            population_sample["min_tail_ess_per_chain"]
            if population_sample is not None
            else None
        ),
        "min_tail_ess_dimension_per_chain": (
            population_sample["min_tail_ess_dimension_per_chain"]
            if population_sample is not None
            else None
        ),
        "bulk_ess_per_dimension_per_chain": (
            population_sample["bulk_ess_per_dimension_per_chain"]
            if population_sample is not None
            else None
        ),
        "tail_ess_per_dimension_per_chain": (
            population_sample["tail_ess_per_dimension_per_chain"]
            if population_sample is not None
            else None
        ),
        "bulk_ess_per_dimension_pooled": (
            population_sample["bulk_ess_per_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "split_rhat_per_dimension_pooled": (
            population_sample["split_rhat_per_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "tail_ess_per_dimension_pooled": (
            population_sample["tail_ess_per_dimension_pooled"]
            if population_sample is not None
            else None
        ),
        "draws_sha256_per_chain": (
            population_sample["draws_sha256_per_chain"]
            if population_sample is not None
            else None
        ),
        "draws_sha256": (
            sampled["draws_sha256"]
            if population_sample is None
            else None
        ),
        "sampling_grads_per_chain": (
            population_sample["sampling_grads_per_chain"]
            if population_sample is not None
            else None
        ),
        "sampling_divergences_per_chain": (
            population_sample["sampling_divergences_per_chain"]
            if population_sample is not None
            else None
        ),
        "sampling_grads_all_chains": (
            sum(population_sample["sampling_grads_per_chain"])
            if population_sample is not None
            else None
        ),
        "sampling_divergences_all_chains": (
            sum(population_sample["sampling_divergences_per_chain"])
            if population_sample is not None
            else None
        ),
        "ess_per_grad_pooled_population": (
            population_sample["min_bulk_ess_pooled"]
            / (
                warmup_grads
                + sum(population_sample["sampling_grads_per_chain"])
            )
            if population_sample is not None
            else None
        ),
        "ess_per_grad_marginal_per_chain": (
            [
                ess / (sc1_warmup_charge + sampling_cost)
                for ess, sampling_cost in zip(
                    population_sample["min_bulk_ess_per_chain"],
                    population_sample["sampling_grads_per_chain"],
                )
            ]
            if population_sample is not None
            else None
        ),
        "population_quality_pass": (
            population_sample["split_rhat_all_finite"]
            and population_sample["max_split_rhat_pooled"] <= 1.01
            and sum(population_sample["sampling_divergences_per_chain"])
            == 0
            if population_sample is not None
            else None
        ),
        "population_quality_rule": (
            "rank_split_rhat<=1.01_and_zero_sampling_divergences"
            if population_sample is not None
            else None
        ),
        "draws_artifact": array_artifact,
        "step_size": float(np.asarray(result.parameters["step_size"])),
        "warmup_wall_seconds": warmup_wall_seconds,
        "sampling_wall_seconds": sampling_wall_seconds,
        "end_to_end_wall_seconds": (
            warmup_wall_seconds + sampling_wall_seconds
        ),
        "bulk_ess_per_second_pooled": (
            population_sample["min_bulk_ess_pooled"]
            / (warmup_wall_seconds + sampling_wall_seconds)
            if population_sample is not None
            else None
        ),
        "dispersion": dispersion_metrics(cell["inits"]),
    }

    events: list[dict[str, Any]] = []
    if capture_events:
        events = extract_schedule_events(
            cell=cell,
            max_grad_budget=max_grad_budget,
            num_warmup_steps=num_warmup_steps,
            integration_steps=integration_steps,
            acceptance=acceptance,
            step_size=step_size,
            scalar_trace=scalar_trace,
            verdict=verdict,
            prescriptions=prescriptions,
        )
    return row, events


def _low_rank_buffer_size(num_steps: int) -> int:
    schedule = np.asarray(build_growing_window_schedule(num_steps))
    ends = np.flatnonzero(schedule[:, 1].astype(bool))
    if ends.size:
        largest_window = int(
            np.max(np.diff(np.concatenate((np.array([-1]), ends))))
        )
    else:
        largest_window = num_steps
    return max(
        min(max(num_steps // 5, 128) * 2, max(num_steps, 1)),
        largest_window + 1,
    )


def run_manual_single_chain(
    cell: Mapping[str, Any],
    *,
    metric: str,
    max_rank: int | None,
    num_warmup_steps: int,
    num_draws: int,
    algorithm: Any = blackjax.nuts,
    algorithm_name: str = "nuts",
    extra_parameters: Mapping[str, Any] | None = None,
    chain_index: int = 0,
    draws_path: Path | None = None,
    return_arrays: bool = False,
) -> dict[str, Any]:
    """Run a paired single-chain manual metric comparator."""

    extra_parameters = dict(extra_parameters or {})
    if metric == "fisher_low_rank":
        if max_rank is None:
            raise ValueError("fisher_low_rank requires max_rank")
        recipe = dataclasses.replace(
            REGISTRY["fisher_low_rank"],
            max_rank=min(max_rank, int(cell["d"])),
        )
        metric_spec: Any = recipe.build_core(
            buffer_size=_low_rank_buffer_size(num_warmup_steps)
        )
        schedule_fn = build_growing_window_schedule

        def info_fn(state: Any, info: Any, adaptation_state: Any) -> tuple:
            return (
                info.num_integration_steps,
                info.is_divergent,
                adaptation_state.step_size,
            )

    elif metric == "welford_diag":
        metric_spec = metric
        schedule_fn = build_growing_window_schedule

        def info_fn(state: Any, info: Any, adaptation_state: Any) -> tuple:
            return (
                info.num_integration_steps,
                info.is_divergent,
                adaptation_state.step_size,
            )

    else:
        raise ValueError(f"Unsupported manual metric {metric!r}")

    warmup = staged_adaptation(
        algorithm,
        cell["logdensity_fn"],
        metric=metric_spec,
        schedule_fn=schedule_fn,
        adaptation_info_fn=info_fn,
        **extra_parameters,
    )
    if not 0 <= chain_index < int(cell["n_chains"]):
        raise ValueError(f"chain_index out of range: {chain_index}")
    initial_position = np.asarray(cell["inits"])[chain_index]
    warmup_key = (
        cell["warmup_key"]
        if chain_index == 0
        else jax.random.fold_in(cell["warmup_key"], chain_index)
    )
    sample_key = (
        cell["sample_key"]
        if chain_index == 0
        else jax.random.fold_in(cell["sample_key"], chain_index)
    )
    started = time.monotonic()
    result, trace = warmup.run(
        warmup_key,
        initial_position,
        num_warmup_steps,
    )
    wall_seconds = time.monotonic() - started
    warmup_nis, warmup_divergence_trace, step_size_trace = trace
    sample_started = time.monotonic()
    sampled = sample_single_chain(
        algorithm,
        cell["logdensity_fn"],
        result.parameters,
        np.asarray(result.state.position),
        sample_key,
        num_draws=num_draws,
        extra_parameters=extra_parameters,
    )
    sampling_wall_seconds = time.monotonic() - sample_started
    warmup_grads = int(np.asarray(warmup_nis).sum())
    total_grads = warmup_grads + sampled["sampling_grads"]
    deployed = result.parameters["inverse_mass_matrix"]
    lam = np.asarray(deployed.lam) if hasattr(deployed, "lam") else np.array([])
    terminal_position = np.asarray(result.state.position)
    arrays = {
        "draws": sampled["draws"],
        "sampling_integration_steps": sampled["sampling_integration_steps"],
        "sampling_divergences": sampled["sampling_divergence_trace"],
        "warmup_integration_steps": np.asarray(warmup_nis),
        "warmup_divergences": np.asarray(warmup_divergence_trace),
        "initial_position": initial_position,
        "warmup_end_position": terminal_position,
    }
    array_artifact = (
        persist_array_artifact(draws_path, arrays)
        if draws_path is not None
        else None
    )
    row = {
        "record_type": "cell",
        "schema_version": SCHEMA_VERSION,
        "model": cell["model"],
        "seed": int(cell["seed"]),
        "dimension": int(cell["d"]),
        "n_chains": 1,
        "algorithm": algorithm_name,
        "initialization_design": (
            f"{cell['initialization_design']}:chain{chain_index}"
        ),
        "chain_index": chain_index,
        "initial_position_sha256": sha256_array(initial_position),
        "warmup_end_position_sha256": sha256_array(terminal_position),
        "warmup_key_sha256": sha256_array(warmup_key),
        "sample_key_sha256": sha256_array(sample_key),
        "metric": metric,
        "max_rank": max_rank,
        "schedule_family": "proportional_growing_reset",
        "num_warmup_steps": num_warmup_steps,
        "num_sampling_draws": num_draws,
        "warmup_grads": warmup_grads,
        "warmup_divergences": int(np.asarray(warmup_divergence_trace).sum()),
        "sampling_grads": sampled["sampling_grads"],
        "sampling_divergences": sampled["sampling_divergences"],
        "min_bulk_ess": sampled["min_bulk_ess"],
        "min_bulk_ess_dimension": sampled["min_bulk_ess_dimension"],
        "bulk_ess_per_dimension": sampled["bulk_ess_per_dimension"],
        "min_tail_ess": sampled["min_tail_ess"],
        "min_tail_ess_dimension": sampled["min_tail_ess_dimension"],
        "tail_ess_per_dimension": sampled["tail_ess_per_dimension"],
        "draws_sha256": sampled["draws_sha256"],
        "ess_per_grad_sc1": sampled["min_bulk_ess"] / total_grads,
        "ess_per_grad_one_output_total": (
            sampled["min_bulk_ess"] / total_grads
        ),
        "draws_artifact": array_artifact,
        "step_size": float(np.asarray(result.parameters["step_size"])),
        "settled_log_step_size_amplitude": float(
            np.ptp(np.log(np.asarray(step_size_trace)[-max(20, num_warmup_steps // 5) :]))
        ),
        "deployed_lam": [float(value) for value in lam],
        "warmup_wall_seconds": wall_seconds,
        "sampling_wall_seconds": sampling_wall_seconds,
        "end_to_end_wall_seconds": wall_seconds + sampling_wall_seconds,
        "bulk_ess_per_second": (
            sampled["min_bulk_ess"]
            / (wall_seconds + sampling_wall_seconds)
        ),
    }
    if return_arrays:
        row["_arrays"] = arrays
    return row


def run_manual_population(
    cell: Mapping[str, Any],
    *,
    metric: str,
    max_rank: int | None,
    nominal_max_grad_budget: int,
    num_warmup_steps_per_chain: int,
    num_draws: int,
    algorithm: Any = blackjax.nuts,
    algorithm_name: str = "nuts",
    extra_parameters: Mapping[str, Any] | None = None,
    draws_path: Path | None = None,
) -> dict[str, Any]:
    """Run the preregistered equal-split eight-chain manual control."""

    chain_rows: list[dict[str, Any]] = []
    chain_arrays: list[dict[str, Any]] = []
    for chain_index in range(int(cell["n_chains"])):
        chain_row = run_manual_single_chain(
            cell,
            metric=metric,
            max_rank=max_rank,
            num_warmup_steps=num_warmup_steps_per_chain,
            num_draws=num_draws,
            algorithm=algorithm,
            algorithm_name=algorithm_name,
            extra_parameters=extra_parameters,
            chain_index=chain_index,
            return_arrays=True,
        )
        chain_arrays.append(chain_row.pop("_arrays"))
        chain_rows.append(chain_row)

    draws = np.stack([arrays["draws"] for arrays in chain_arrays])
    pooled_bulk_ess = np.asarray(az.ess(draws, method="bulk"))
    pooled_tail_ess = np.asarray(
        az.ess(draws, method="tail", prob=(0.05, 0.95))
    )
    pooled_split_rhat = np.asarray(az.rhat(draws, method="rank"))
    rhat_is_finite = bool(np.isfinite(pooled_split_rhat).all())
    warmup_grads_per_chain = [
        int(row["warmup_grads"]) for row in chain_rows
    ]
    sampling_grads_per_chain = [
        int(row["sampling_grads"]) for row in chain_rows
    ]
    warmup_divergences_per_chain = [
        int(row["warmup_divergences"]) for row in chain_rows
    ]
    sampling_divergences_per_chain = [
        int(row["sampling_divergences"]) for row in chain_rows
    ]
    warmup_grads_all_chains = sum(warmup_grads_per_chain)
    sampling_grads_all_chains = sum(sampling_grads_per_chain)
    sampling_divergences_all_chains = sum(
        sampling_divergences_per_chain
    )
    total_wall_seconds = sum(
        float(row["end_to_end_wall_seconds"]) for row in chain_rows
    )
    min_bulk_ess_pooled = float(np.min(pooled_bulk_ess))
    min_tail_ess_pooled = float(np.min(pooled_tail_ess))
    array_artifact = (
        persist_array_artifact(
            draws_path,
            {
                "draws": draws,
                "sampling_integration_steps": np.stack(
                    [
                        arrays["sampling_integration_steps"]
                        for arrays in chain_arrays
                    ]
                ),
                "sampling_divergences": np.stack(
                    [
                        arrays["sampling_divergences"]
                        for arrays in chain_arrays
                    ]
                ),
                "warmup_integration_steps": np.stack(
                    [
                        arrays["warmup_integration_steps"]
                        for arrays in chain_arrays
                    ]
                ),
                "warmup_divergences": np.stack(
                    [
                        arrays["warmup_divergences"]
                        for arrays in chain_arrays
                    ]
                ),
                "initial_positions": np.stack(
                    [
                        arrays["initial_position"]
                        for arrays in chain_arrays
                    ]
                ),
                "warmup_end_positions": np.stack(
                    [
                        arrays["warmup_end_position"]
                        for arrays in chain_arrays
                    ]
                ),
            },
        )
        if draws_path is not None
        else None
    )
    total_gradients = warmup_grads_all_chains + sampling_grads_all_chains
    return {
        "record_type": "cell",
        "schema_version": SCHEMA_VERSION,
        "model": cell["model"],
        "seed": int(cell["seed"]),
        "dimension": int(cell["d"]),
        "n_chains": int(cell["n_chains"]),
        "algorithm": algorithm_name,
        "comparison_policy": "equal_split_aggregate_budget_control",
        "metric_analysis_role": (
            "preregistered_primary_manual_metric"
            if metric == "fisher_low_rank"
            else "preregistered_diagonal_control"
        ),
        "initialization_design": cell["initialization_design"],
        "initial_positions_sha256": sha256_array(cell["inits"]),
        "metric": metric,
        "max_rank": max_rank,
        "schedule_family": "proportional_growing_reset",
        "nominal_max_grad_budget": nominal_max_grad_budget,
        "budget_enforcement": (
            "same_preregistered_nominal_allocation_not_realized_cap"
        ),
        "warmup_step_conversion_assumed_integration_steps": (
            _ASSUMED_AVG_LEAPFROGS_PER_STEP
        ),
        "num_warmup_steps_per_chain": num_warmup_steps_per_chain,
        "nominal_aggregate_warmup_transitions": (
            int(cell["n_chains"]) * num_warmup_steps_per_chain
        ),
        "nominal_unused_gradient_remainder": (
            nominal_max_grad_budget
            - int(cell["n_chains"])
            * num_warmup_steps_per_chain
            * _ASSUMED_AVG_LEAPFROGS_PER_STEP
        ),
        "num_sampling_draws_per_chain": num_draws,
        "warmup_grads_per_chain": warmup_grads_per_chain,
        "warmup_grads_all_chains": warmup_grads_all_chains,
        "warmup_divergences_per_chain": warmup_divergences_per_chain,
        "warmup_divergences_all_chains": sum(
            warmup_divergences_per_chain
        ),
        "sampling_grads_per_chain": sampling_grads_per_chain,
        "sampling_grads_all_chains": sampling_grads_all_chains,
        "sampling_divergences_per_chain": (
            sampling_divergences_per_chain
        ),
        "sampling_divergences_all_chains": (
            sampling_divergences_all_chains
        ),
        "min_bulk_ess_per_chain": [
            row["min_bulk_ess"] for row in chain_rows
        ],
        "min_bulk_ess_dimension_per_chain": [
            row["min_bulk_ess_dimension"] for row in chain_rows
        ],
        "bulk_ess_per_dimension_per_chain": [
            row["bulk_ess_per_dimension"] for row in chain_rows
        ],
        "min_tail_ess_per_chain": [
            row["min_tail_ess"] for row in chain_rows
        ],
        "min_tail_ess_dimension_per_chain": [
            row["min_tail_ess_dimension"] for row in chain_rows
        ],
        "tail_ess_per_dimension_per_chain": [
            row["tail_ess_per_dimension"] for row in chain_rows
        ],
        "min_bulk_ess_pooled": min_bulk_ess_pooled,
        "min_bulk_ess_dimension_pooled": int(
            np.argmin(pooled_bulk_ess)
        ),
        "bulk_ess_per_dimension_pooled": pooled_bulk_ess.tolist(),
        "min_tail_ess_pooled": min_tail_ess_pooled,
        "min_tail_ess_dimension_pooled": int(
            np.argmin(pooled_tail_ess)
        ),
        "tail_ess_per_dimension_pooled": pooled_tail_ess.tolist(),
        "max_split_rhat_pooled": (
            float(np.max(pooled_split_rhat)) if rhat_is_finite else None
        ),
        "max_split_rhat_dimension_pooled": (
            int(np.argmax(pooled_split_rhat)) if rhat_is_finite else None
        ),
        "split_rhat_per_dimension_pooled": [
            finite_or_none(value) for value in pooled_split_rhat
        ],
        "split_rhat_all_finite": rhat_is_finite,
        "ess_per_grad_marginal_per_chain": [
            row["ess_per_grad_sc1"] for row in chain_rows
        ],
        "ess_per_grad_pooled_population": (
            min_bulk_ess_pooled / total_gradients
        ),
        "population_quality_pass": (
            rhat_is_finite
            and float(np.max(pooled_split_rhat)) <= 1.01
            and sampling_divergences_all_chains == 0
        ),
        "population_quality_rule": (
            "rank_split_rhat<=1.01_and_zero_sampling_divergences"
        ),
        "warmup_key_sha256_per_chain": [
            row["warmup_key_sha256"] for row in chain_rows
        ],
        "sample_key_sha256_per_chain": [
            row["sample_key_sha256"] for row in chain_rows
        ],
        "warmup_end_position_sha256_per_chain": [
            row["warmup_end_position_sha256"] for row in chain_rows
        ],
        "draws_sha256_per_chain": [
            row["draws_sha256"] for row in chain_rows
        ],
        "step_size_per_chain": [row["step_size"] for row in chain_rows],
        "deployed_lam_per_chain": [
            row["deployed_lam"] for row in chain_rows
        ],
        "settled_log_step_size_amplitude_per_chain": [
            row["settled_log_step_size_amplitude"]
            for row in chain_rows
        ],
        "warmup_wall_seconds_per_chain": [
            row["warmup_wall_seconds"] for row in chain_rows
        ],
        "sampling_wall_seconds_per_chain": [
            row["sampling_wall_seconds"] for row in chain_rows
        ],
        "end_to_end_wall_seconds_sequential": total_wall_seconds,
        "bulk_ess_per_second_sequential": (
            min_bulk_ess_pooled / total_wall_seconds
        ),
        "draws_artifact": array_artifact,
    }


def extract_schedule_events(
    *,
    cell: Mapping[str, Any],
    max_grad_budget: int,
    num_warmup_steps: int,
    integration_steps: Any,
    acceptance: Any,
    step_size: Any,
    scalar_trace: Sequence[Any],
    verdict: Any,
    prescriptions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Persist one row per prescribed controller metric-window boundary."""

    acceptance_arr = np.asarray(acceptance)
    traces = {
        name: np.asarray(values)
        for name, values in zip(MC_SCALAR_NAMES, scalar_trace)
    }

    prior_escalated = False
    prior_handoff = "none"
    branch_names = {
        _DETECTION_BRANCH_NONE: "none",
        _DETECTION_BRANCH_POOLED_WITHIN: "pooled_within",
        _DETECTION_BRANCH_BETWEEN_MEANS: "between_means",
        _DETECTION_BRANCH_BOTH: "both",
    }

    # ``verdict`` is intentionally not used to relabel prior windows: each
    # event reports evidence available at its own prescribed boundary.
    del verdict

    def _evidence(
        trace_index: int,
        window: Mapping[str, int],
    ) -> Mapping[str, Any]:
        nonlocal prior_escalated, prior_handoff

        escalated = bool(traces["has_escalated"][trace_index])
        deferred = bool(traces["deferred_to_ensemble"][trace_index])
        r2 = finite_or_none(traces["r2_latest"][trace_index])
        branch = branch_names.get(
            int(traces["detection_branch"][trace_index]),
            "unknown",
        )
        if deferred:
            handoff = "population"
        elif not escalated and r2 is not None and r2 < _R_MIN:
            handoff = "reparameterize"
        else:
            handoff = "none"
        transition: list[str] = []
        if escalated and not prior_escalated:
            transition.append("diagonal_to_low_rank")
        if handoff != "none" and handoff != prior_handoff:
            transition.append(f"handoff_{handoff}")
        converged_at = int(traces["converged_at_step"][trace_index])
        start_step = int(window["window_start_step"])
        end_step = int(window["window_end_step"])
        if start_step <= converged_at <= end_step:
            transition.append("airm_stable")
        prior_escalated = escalated
        prior_handoff = handoff
        return {
            "record_type": "schedule_event",
            "schema_version": SCHEMA_VERSION,
            "model": cell["model"],
            "seed": int(cell["seed"]),
            "max_grad_budget": max_grad_budget,
            "schedule_family": prescriptions["controller_actual"]["name"],
            "stage": "slow",
            "end_fraction": float(end_step / num_warmup_steps),
            "mean_acceptance": float(
                np.mean(acceptance_arr[start_step - 1 : end_step])
            ),
            "route": "low_rank" if escalated else "diagonal",
            "effective_rank": (
                int(traces["escalation_rank"][trace_index])
                if escalated
                else 0
            ),
            "route_latched": "low_rank" if escalated else "diagonal",
            "metric_route_basis": branch if escalated else "none",
            "handoff": handoff,
            "transition_events": transition,
            "nominal_rank": int(
                traces["escalation_rank"][trace_index]
            ),
            "within_lam1": finite_or_none(
                traces["within_lam1"][trace_index]
            ),
            "chain_consistency_psi": finite_or_none(
                traces["chain_consistency_psi"][trace_index]
            ),
            "chain_collinearity": finite_or_none(
                traces["chain_collinearity"][trace_index]
            ),
            "r2_latest": r2,
            "airm_velocity": finite_or_none(
                traces["airm_vel_curr"][trace_index]
            ),
            "airm_converged_at_step": (
                converged_at if converged_at >= 0 else None
            ),
        }

    return build_metric_window_events(
        prescriptions=prescriptions,
        integration_steps=integration_steps,
        step_sizes=step_size,
        controller_budget_trace=traces["budget_used"],
        evidence_at_boundary=_evidence,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_unique_cells(
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("record_type") != "cell":
            continue
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            errors.append(f"duplicate cell key {key}")
        seen.add(key)
        if row.get("error"):
            errors.append(f"cell {key} has error: {row['error']}")
    return errors


def finish_output(path: Path) -> dict[str, Any]:
    return {
        "path": Path(path).name,
        "sha256": sha256_file(path),
        "bytes": Path(path).stat().st_size,
    }


__all__ = [
    "CURRENT_BLACKJAX_SHA",
    "HISTORICAL_SEQUENTIAL_SHA",
    "ImmutableJsonl",
    "SCHEMA_VERSION",
    "TUNINGFORK_SHA",
    "build_isotropic_cell",
    "build_multi_chain_cell",
    "canonical_json",
    "finish_output",
    "load_jsonl",
    "persist_array_artifact",
    "provenance",
    "run_auto_multi_chain",
    "run_manual_population",
    "run_manual_single_chain",
    "sample_chain_population",
    "sample_single_chain",
    "sha256_array",
    "sha256_file",
    "validate_unique_cells",
]
