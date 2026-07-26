"""5D GMM metric-fixable→reparam boundary experiment.

Constructs a 5D bimodal Gaussian mixture with fixed marginal covariance (SR=0
is unimodal; SR→10 is strongly bimodal) and runs the merged-main multi-chain
meta-adaptation controller (metric='auto', M=8) at each separation ratio SR.

The marginal covariance Σ_marginal = I5 + a·vvᵀ is INVARIANT across SR by
construction (the within/between covariance split absorbs the separation).
This lets the experiment isolate the controller's multimodality detection from
its elongation detection.

Usage
-----
  python experiments/gmm_boundary.py --smoke --out smoke.jsonl
  python experiments/gmm_boundary.py --sweep --out results.jsonl
  python experiments/gmm_boundary.py --single-chain --out single-chain.jsonl

Environment
-----------
PYTHONPATH=<BlackJAX source checkout>
Python environment containing JAX, ArviZ, NumPy, and Matplotlib.
x64 required: jax.config.update("jax_enable_x64", True)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

# x64 is mandatory: the low-rank estimator is numerically indefinite under f32.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# ---------------------------------------------------------------------------
# Construction constants (exact spec from the task)
# ---------------------------------------------------------------------------
_PI1: float = 0.30  # weight of component 1
_PI2: float = 0.70  # weight of component 2
_D: int = 5
_A: float = 21.0  # elongation scalar; Sigma_marginal = I5 + a·vvᵀ

# ---------------------------------------------------------------------------
# GMMSpec
# ---------------------------------------------------------------------------


class GMMSpec(NamedTuple):
    """Specification of a 5D Gaussian mixture at a given separation ratio SR."""

    mu1: np.ndarray  # (5,) mean of component 1 (weight π1=0.30)
    mu2: np.ndarray  # (5,) mean of component 2 (weight π2=0.70)
    Sigma_within: np.ndarray  # (5,5) within-mode covariance (shared across components)
    Sigma_marginal: np.ndarray  # (5,5) marginal covariance (INVARIANT across SR)
    delta: float  # separation scale parameter
    beta: float  # within-mode elongation: Sigma_within = I5 + β·vvᵀ
    correlated_axes: int  # number of leading coordinates in the correlated direction
    logdensity_fn: Callable[..., Any]  # closed over jnp arrays


# ---------------------------------------------------------------------------
# 1. build_gmm
# ---------------------------------------------------------------------------


def _correlated_direction(correlated_axes: int) -> np.ndarray:
    """Return the unit direction supported on ``correlated_axes`` coordinates."""
    if isinstance(correlated_axes, bool) or not isinstance(
        correlated_axes, (int, np.integer)
    ) or not 2 <= correlated_axes <= _D:
        raise ValueError(
            f"correlated_axes must be between 2 and {_D}, got {correlated_axes}."
        )
    v = np.zeros(_D)
    v[:correlated_axes] = 1.0 / math.sqrt(correlated_axes)
    return v


def analytic_diagnostic(correlated_axes: int = 2) -> dict[str, float | int]:
    """Return exact marginal geometry diagnostics without constructing a sampler.

    The diagonal-whitened marginal has one spike
    ``1 + (k - 1) * a / (k + a)`` and ``k choose 2`` nonzero off-diagonals.
    """
    _correlated_direction(correlated_axes)
    k = int(correlated_axes)
    return {
        "correlated_axes": k,
        "marginal_whitened_spike": 1.0 + (k - 1) * _A / (k + _A),
        "off_diagonal_correlations": k * (k - 1) // 2,
    }


def correlated_projection(
    positions: np.ndarray, correlated_axes: int = 2
) -> np.ndarray:
    """Project positions onto the known correlated mixture direction."""
    return np.asarray(positions) @ _correlated_direction(correlated_axes)


def build_gmm(SR: float, correlated_axes: int = 2) -> GMMSpec:
    """Build a 5D Gaussian mixture with invariant marginal covariance.

    Parameters
    ----------
    SR
        Separation ratio in [0, 10].  SR=0 is unimodal (β=a); SR=10 has
        spherical within-mode covariance (β=0).

    Returns
    -------
    GMMSpec
        Construction fields + a JAX-differentiable logdensity_fn.
    """
    pi1, pi2 = _PI1, _PI2
    a = _A
    v = _correlated_direction(correlated_axes)

    Sigma_marginal = np.eye(_D) + a * np.outer(v, v)

    # δ² chosen so Σ_within + π1π2δ²vvᵀ = Σ_marginal exactly.
    delta_sq = SR**2 * (1.0 + a) / (1.0 + pi1 * pi2 * SR**2)
    delta = float(np.sqrt(delta_sq))

    # β = a − π1π2δ²; β≥0 for SR≤10 (SR=10 → β=0, SR>10 would give β<0).
    beta = a - pi1 * pi2 * delta_sq

    Sigma_within = np.eye(_D) + beta * np.outer(v, v)

    # Grand mean = π1μ1 + π2μ2 = π1(−π2δv) + π2(π1δv) = 0 ✓
    mu1 = -pi2 * delta * v  # = −0.70·δ·v
    mu2 = pi1 * delta * v  # = +0.30·δ·v

    # Precompute Cholesky for log-density evaluation (closed over in logdensity_fn).
    L_within = np.linalg.cholesky(Sigma_within)
    L_inv = np.linalg.inv(L_within)
    # log det(2π Σ_within) = D·log(2π) + log det(Σ_within) = D·log(2π) + 2·Σ log L_ii
    log_det_term = float(
        _D * np.log(2 * np.pi) + 2.0 * np.sum(np.log(np.diag(L_within)))
    )

    # Convert to JAX arrays (f64 already set globally).
    mu1_j = jnp.array(mu1)
    mu2_j = jnp.array(mu2)
    L_inv_j = jnp.array(L_inv)
    log_pi1 = float(np.log(pi1))
    log_pi2 = float(np.log(pi2))

    def logdensity_fn(x: jax.Array) -> jax.Array:
        """log p(x) = logsumexp_k[ log π_k − ½ (x−μk)ᵀ Σ_within⁻¹ (x−μk) − ½ log det(2π Σ) ]."""

        def _comp(mu_j: jax.Array, log_pi: float) -> jax.Array:
            z = L_inv_j @ (x - mu_j)
            return log_pi - 0.5 * jnp.dot(z, z) - 0.5 * log_det_term

        lp1 = _comp(mu1_j, log_pi1)
        lp2 = _comp(mu2_j, log_pi2)
        return logsumexp(jnp.array([lp1, lp2]))

    return GMMSpec(
        mu1=mu1,
        mu2=mu2,
        Sigma_within=Sigma_within,
        Sigma_marginal=Sigma_marginal,
        delta=delta,
        beta=beta,
        correlated_axes=correlated_axes,
        logdensity_fn=logdensity_fn,
    )


# ---------------------------------------------------------------------------
# 2. verify_invariance
# ---------------------------------------------------------------------------


def verify_invariance(
    SR_grid: np.ndarray | list[float],
    correlated_axes: int = 2,
) -> tuple[float, float, list[float]]:
    """Verify that Σ_within + π1π2δ²vvᵀ = Σ_marginal over SR_grid.

    Parameters
    ----------
    SR_grid
        Iterable of SR values to check.
    correlated_axes
        Number ``k`` of equally supported correlated axes. The
        diagonal-whitened marginal spike is ``(1 + a) / (1 + a / k)``:
        approximately ``1.913`` for the default ``k=2`` and ``2.750`` for
        ``k=3``.

    Returns
    -------
    max_err
        max ‖Σ_within + π1π2δ²vvᵀ − Σ_marginal‖∞ over the grid.
        **Must be < 1e-12** (any value above signals a construction error).
    lam1_marginal
        Top eigenvalue of D(Σ_marginal)^{−½} Σ_marginal D(Σ_marginal)^{−½}.
        Constant over SR, with its value determined by ``correlated_axes``.
    lam1_within_grid
        Top eigenvalue of D(Σ_within)^{−½} Σ_within D(Σ_within)^{−½} at each SR.
        Starts at the corresponding marginal spike (SR=0) and decreases toward
        1.0 as the within-mode structure becomes isotropic.
    """
    pi1, pi2 = _PI1, _PI2
    a = _A
    v = _correlated_direction(correlated_axes)

    Sigma_marginal = np.eye(_D) + a * np.outer(v, v)

    # Marginal whitened λ1: constant over SR.
    D_marg = np.diag(np.diag(Sigma_marginal))
    D_marg_invsqrt = np.diag(1.0 / np.sqrt(np.diag(D_marg)))
    whitened_marg = D_marg_invsqrt @ Sigma_marginal @ D_marg_invsqrt
    lam1_marginal = float(np.max(np.linalg.eigvalsh(whitened_marg)))

    max_err = 0.0
    lam1_within_grid: list[float] = []

    for SR in SR_grid:
        spec = build_gmm(float(SR), correlated_axes)

        # Invariance check.
        delta_sq = spec.delta**2
        reconstruction = spec.Sigma_within + pi1 * pi2 * delta_sq * np.outer(v, v)
        err = float(np.max(np.abs(reconstruction - Sigma_marginal)))
        max_err = max(max_err, err)

        # Within-mode whitened λ1.
        D_w = np.diag(np.diag(spec.Sigma_within))
        D_w_invsqrt = np.diag(1.0 / np.sqrt(np.diag(D_w)))
        whitened_w = D_w_invsqrt @ spec.Sigma_within @ D_w_invsqrt
        lam1_w = float(np.max(np.linalg.eigvalsh(whitened_w)))
        lam1_within_grid.append(lam1_w)

    return max_err, lam1_marginal, lam1_within_grid


# ---------------------------------------------------------------------------
# 3. projected_transcript_decomposition
# ---------------------------------------------------------------------------


def projected_transcript_decomposition(
    warmup_positions: np.ndarray,
    spec: GMMSpec,
) -> dict[str, float]:
    """Return the exact equal-length W/B/S decomposition along the declared v.

    ``warmup_positions`` is ``warmup_info.state.position`` from a multi-chain
    run, with shape ``(n_steps, n_chains, d)``.  The calculation is
    evaluation-only and is not supplied to any controller gate.

    The reported partition error is signed and remains in variance units:
    ``S - {M(n-1)W + (M-1)B}/(Mn-1)``.  The reference error is likewise
    ``S - v.T @ Sigma_marginal @ v``.
    """
    positions = np.asarray(warmup_positions)
    if positions.ndim != 3:
        raise ValueError(
            "warmup_positions must have shape (n_steps, n_chains, d), "
            f"got {positions.shape}."
        )

    n, M, d = positions.shape
    if n < 2 or M < 2:
        raise ValueError(
            "projected W/B/S diagnostics require at least two steps and two "
            f"chains, got n_steps={n}, n_chains={M}."
        )
    if d != spec.Sigma_marginal.shape[0]:
        raise ValueError(
            "warmup position dimension does not match the GMM specification: "
            f"{d} != {spec.Sigma_marginal.shape[0]}."
        )

    v = _correlated_direction(spec.correlated_axes)
    projected = np.einsum("tmd,d->tm", positions, v)
    chain_means = np.mean(projected, axis=0)
    grand_mean = float(np.mean(projected))

    within = float(
        np.sum((projected - chain_means[None, :]) ** 2) / (M * (n - 1))
    )
    between = float(n * np.sum((chain_means - grand_mean) ** 2) / (M - 1))
    total = float(np.sum((projected - grand_mean) ** 2) / (M * n - 1))
    target = float(v @ spec.Sigma_marginal @ v)

    reconstructed_total = (
        M * (n - 1) * within + (M - 1) * between
    ) / (M * n - 1)

    return {
        "projected_within_variance": within,
        "projected_between_variance": between,
        "projected_total_variance": total,
        "projected_target_variance": target,
        "projected_partition_error": total - reconstructed_total,
        "projected_total_reference_error": total - target,
    }


# ---------------------------------------------------------------------------
# 4. init_positions
# ---------------------------------------------------------------------------


def init_positions(
    SR: float,
    M: int,
    seed: int,
    kind: str,
    correlated_axes: int = 2,
) -> np.ndarray:
    """Generate M initial positions of shape (M, 5).

    Parameters
    ----------
    SR
        Separation ratio used to determine mode locations.
    M
        Number of chains.
    seed
        NumPy random seed.
    kind
        One of ``"broad"``, ``"mode_centered"``, ``"split"``.

    Returns
    -------
    Array of shape (M, 5).
    """
    rng = np.random.default_rng(seed)
    spec = build_gmm(SR, correlated_axes)

    if kind == "broad":
        # N(0, 2·Σ_marginal): overdispersed starts covering the full marginal.
        L_marg = np.linalg.cholesky(2.0 * spec.Sigma_marginal)
        return rng.standard_normal((M, _D)) @ L_marg.T

    elif kind == "mode_centered":
        # Oracle: draw each chain near one of the two modes with prob proportional
        # to mixture weights.
        L_w = np.linalg.cholesky(spec.Sigma_within)
        positions = np.zeros((M, _D))
        for i in range(M):
            if rng.random() < _PI1:
                positions[i] = spec.mu1 + rng.standard_normal(_D) @ L_w.T
            else:
                positions[i] = spec.mu2 + rng.standard_normal(_D) @ L_w.T
        return positions

    elif kind == "split":
        # Deterministic: ceil(π2·M) chains near μ2, remaining near μ1.
        # Controls start dispersion: fraction near each mode matches the true
        # weight, providing a robust robustness baseline.
        n2 = math.ceil(_PI2 * M)
        n1 = M - n2
        L_w = np.linalg.cholesky(spec.Sigma_within)
        pos_mu1 = spec.mu1[None, :] + rng.standard_normal((n1, _D)) @ L_w.T
        pos_mu2 = spec.mu2[None, :] + rng.standard_normal((n2, _D)) @ L_w.T
        return np.vstack([pos_mu1, pos_mu2])

    else:
        raise ValueError(
            f"Unknown init kind '{kind}'. Must be broad, mode_centered, or split."
        )


# ---------------------------------------------------------------------------
# 5. run_point
# ---------------------------------------------------------------------------


def run_point(
    SR: float,
    seed: int,
    init_kind: str,
    budget: int,
    M: int = 8,
    n_sample_draws: int = 1000,
    correlated_axes: int = 2,
) -> dict:
    """Run the multi-chain meta-adaptation controller on the 5D GMM at SR.

    Parameters
    ----------
    SR
        Separation ratio.
    seed
        JAX random seed.
    init_kind
        One of ``"broad"``, ``"mode_centered"``, ``"split"``.
    budget
        Maximum gradient budget (max_grad_budget for staged_adaptation).
    M
        Number of chains (default 8).
    n_sample_draws
        Post-warmup draws per chain.

    Returns
    -------
    dict
        Flat dict of observables.  An ``"error"`` field is added (and all
        numeric fields set to None) if an exception occurs — never raises.
    """
    # Import here so the module is importable before blackjax is available.
    import blackjax
    from blackjax.adaptation.meta import extract_multi_chain_verdict
    from blackjax.adaptation.staged_adaptation import staged_adaptation

    out: dict = {
        "SR": float(SR),
        "correlated_axes": int(correlated_axes),
        "seed": int(seed),
        "init_kind": str(init_kind),
        "budget": int(budget),
        "M": int(M),
        "n_sample_draws": int(n_sample_draws),
        # Controller verdict fields
        "route": None,
        "effective_rank": None,
        "deferred_to_ensemble": None,
        "detection_branch": None,
        "within_lam1": None,
        "chain_consistency_psi": None,
        "chain_collinearity": None,
        "unimodality_gate": None,
        "mode_coverage": None,
        "confidence": None,
        # Corrected verdict semantics (post-repair calibration pending)
        "metric_route_status": None,
        "metric_route_basis": None,
        "metric_scope": None,
        "observed_ensemble_evidence": None,
        "global_exploration": None,
        "handoff": None,
        "confidence_scope": None,
        # Exact warmup-transcript W/B/S diagnostics along the declared v
        "projected_within_variance": None,
        "projected_between_variance": None,
        "projected_total_variance": None,
        "projected_target_variance": None,
        "projected_partition_error": None,
        "projected_total_reference_error": None,
        # Recoverable evidence trace at prescribed metric-window boundaries
        "window_events": None,
        "schedule_prescriptions": None,
        # Fields not exposed in MultiChainMetaAdaptationCoreState
        "contraction_t": None,  # not available in v1; recorded as None
        "mode_flag": None,  # not available in v1; recorded as None
        # Post-warmup sampling observables
        "min_ess_per_grad": None,
        "split_rhat": None,
        "mode_weight_est": None,
        "both_modes_visited_frac": None,
        "num_divergences": None,
        "error": None,
    }

    try:
        spec = build_gmm(SR, correlated_axes)
        logdensity_fn = spec.logdensity_fn

        # Initial positions: shape (M, 5)
        positions_np = init_positions(SR, M, seed, init_kind, correlated_axes)
        positions = jnp.array(positions_np)

        rng = jax.random.key(seed)
        warmup_key, sample_key = jax.random.split(rng)

        # --- Warmup ---
        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=budget,
            n_chains=M,
        )
        results, warmup_info = warmup.run(warmup_key, positions)

        from window_events import extract_window_events

        window_events, schedules = extract_window_events(
            warmup_info,
            max_grad_budget=budget,
            n_chains=M,
            dimension=_D,
        )
        out["window_events"] = window_events
        out["schedule_prescriptions"] = schedules

        # Evaluation-only decomposition of the realized warmup transcript.
        # This is deliberately downstream of warmup.run and is never supplied
        # to the controller.
        out.update(
            projected_transcript_decomposition(
                np.asarray(warmup_info.state.position),
                spec,
            )
        )

        # Extract final MultiChainMetaAdaptationCoreState from stacked scan output.
        final_imm_state = jax.tree_util.tree_map(
            lambda x: x[-1], warmup_info.adaptation_state.imm_state
        )

        # num_warmup_steps: use budget_used from the final state (counts warmup steps).
        num_warmup_steps = int(np.asarray(final_imm_state.budget_used))

        verdict = extract_multi_chain_verdict(
            final_imm_state,
            max_grad_budget=budget,
            num_warmup_steps=num_warmup_steps,
        )

        # Populate controller fields from verdict.
        out["route"] = verdict.route
        out["effective_rank"] = int(verdict.effective_rank)
        out["confidence"] = verdict.confidence

        flags = verdict.flags
        out["deferred_to_ensemble"] = bool(flags.get("deferred_to_ensemble", False))
        out["detection_branch"] = str(flags.get("detection_branch", "unknown"))
        out["within_lam1"] = float(flags.get("within_lam1", float("nan")))
        out["chain_consistency_psi"] = float(
            flags.get("chain_consistency_psi", float("nan"))
        )
        out["chain_collinearity"] = float(flags.get("chain_collinearity", float("nan")))
        out["unimodality_gate"] = str(flags.get("unimodality_gate", "unknown"))
        out["mode_coverage"] = str(flags.get("mode_coverage", "unknown"))
        out["metric_route_status"] = str(
            flags.get("metric_route_status", "unassessed")
        )
        out["metric_route_basis"] = str(
            flags.get("metric_route_basis", "none")
        )
        out["metric_scope"] = str(flags.get("metric_scope", "unassessed"))
        out["observed_ensemble_evidence"] = str(
            flags.get("observed_ensemble_evidence", "unassessed")
        )
        out["global_exploration"] = str(
            flags.get("global_exploration", "not_established")
        )
        out["handoff"] = str(flags.get("handoff", "none"))
        out["confidence_scope"] = str(
            flags.get(
                "confidence_scope",
                "historical_route_selection_heuristic",
            )
        )

        # contraction_t and mode_flag are not fields of MultiChainMetaAdaptationCoreState;
        # record None (do not fabricate values from unrelated fields).
        out["contraction_t"] = None
        out["mode_flag"] = None

        # --- Post-warmup NUTS sampling ---
        step_size = results.parameters["step_size"]
        inverse_mass_matrix = results.parameters["inverse_mass_matrix"]
        nuts = blackjax.nuts(
            logdensity_fn, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
        )

        # vmap NUTS step over M chains; scan over draws.
        init_states = results.state  # batched NutsState, shape (M, ...)

        def one_sample_step(states, key):
            chain_keys = jax.random.split(key, M)
            new_states, infos = jax.vmap(nuts.step)(chain_keys, states)
            return new_states, (new_states, infos)

        sample_keys = jax.random.split(sample_key, n_sample_draws)
        _, (all_states, all_infos) = jax.lax.scan(
            one_sample_step, init_states, sample_keys
        )

        # all_states.position shape: (n_draws, M, d)
        positions_draws = np.asarray(all_states.position)  # (n_draws, M, d)
        n_draws_actual, M_actual, d = positions_draws.shape

        # split_rhat: arviz expects (chain, draw) arrays per parameter.
        import arviz as az

        # positions_draws shape: (n_draws, M, d) → transpose to (M, n_draws, d)
        draws_per_chain = np.transpose(positions_draws, (1, 0, 2))  # (M, n_draws, d)

        # Compute rhat per dimension (pass 2D slice (M, n_draws) → scalar rhat).
        rhat_vals = []
        for i in range(d):
            # az.rhat on a 2D (chain, draw) array returns a scalar DataArray.
            rhat_i = float(az.rhat({"x": draws_per_chain[:, :, i]})["x"])
            rhat_vals.append(rhat_i)
        # nanmax: rhat is NaN when a chain has zero variance; report NaN (informative).
        out["split_rhat"] = float(np.nanmax(rhat_vals)) if rhat_vals else float("nan")

        # ESS (bulk) per dimension; per grad.
        total_grads = int(np.asarray(all_infos.num_integration_steps).sum())
        ess_vals = []
        for i in range(d):
            ess_i = float(az.ess({"x": draws_per_chain[:, :, i]})["x"])
            ess_vals.append(ess_i)
        # nanmin: ESS is NaN for a constant chain; floor to 0 (not nan) so the
        # sweep JSONL gets a real number rather than being dropped.
        min_ess_raw = float(np.nanmin(ess_vals)) if ess_vals else 0.0
        min_ess = 0.0 if math.isnan(min_ess_raw) else min_ess_raw
        out["min_ess_per_grad"] = min_ess / max(total_grads, 1)

        # mode_weight_est: fraction of draws nearest to μ2 (true = 0.70).
        mu2_arr = spec.mu2  # (d,)
        mu1_arr = spec.mu1  # (d,)
        flat_draws = positions_draws.reshape(-1, d)  # (n_draws*M, d)
        dist1 = np.sum((flat_draws - mu1_arr) ** 2, axis=1)
        dist2 = np.sum((flat_draws - mu2_arr) ** 2, axis=1)
        near_mu2 = dist2 < dist1
        mode_wt = float(np.mean(near_mu2))
        # At SR=0 both modes coincide; mean(near_mu2) = 0.0 (not nan) but guard anyway.
        out["mode_weight_est"] = 0.0 if math.isnan(mode_wt) else mode_wt

        # both_modes_visited_frac: fraction of chains that visit BOTH modes
        # at least once (a chain "visits" a mode if any draw is nearest that mode).
        # NaN guard: if positions have nan (divergent chain), comparisons return False
        # so the result is 0.0 not nan.  Still add explicit guard for safety.
        chain_has_mu1 = np.zeros(M_actual, dtype=bool)
        chain_has_mu2 = np.zeros(M_actual, dtype=bool)
        for c in range(M_actual):
            chain_draws = draws_per_chain[c]  # (n_draws, d)
            d1 = np.sum((chain_draws - mu1_arr) ** 2, axis=1)
            d2 = np.sum((chain_draws - mu2_arr) ** 2, axis=1)
            chain_has_mu1[c] = bool(np.any(d1 < d2))
            chain_has_mu2[c] = bool(np.any(d2 < d1))
        both = chain_has_mu1 & chain_has_mu2
        both_raw = float(np.mean(both))
        out["both_modes_visited_frac"] = 0.0 if math.isnan(both_raw) else both_raw

        # num_divergences: total divergences across all chains and draws.
        # NUTS doesn't have a divergence field directly exposed in infos;
        # use is_divergent if available, else fall back to None.
        if hasattr(all_infos, "is_divergent"):
            out["num_divergences"] = int(np.sum(np.asarray(all_infos.is_divergent)))
        else:
            out["num_divergences"] = None  # not exposed in this NUTS info version

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return out


# ---------------------------------------------------------------------------
# 5. run_ablation — clean same-warmup metric ablation
# ---------------------------------------------------------------------------


def run_ablation(
    SR: float,
    seed: int,
    budget: int,
    M: int = 8,
    n_sample_draws: int = 1000,
    correlated_axes: int = 2,
) -> dict:
    """Clean metric ablation: deployed low-rank vs diagonal at equal chain count.

    Runs a SINGLE auto warmup (M chains), then branches into two post-warmup
    sampling arms that share the same starting states, step size, and chain
    count — the ONLY difference is the mass matrix:

    * ``arm_lr``   — M-chain NUTS with the deployed
      :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix` (sigma, U, lam).
    * ``arm_diag`` — M-chain NUTS with the *diagonal* of that same IMM, i.e.
      a 1-D array ``sigma**2 * (1 + sum_k (lam_k - 1) * U[:,k]**2)``,
      so only the per-coordinate scaling is retained.

    The paired output distinguishes a measurable low-rank benefit
    (``delta_ess > 0`` or ``delta_modes > 0``) from a route with no matched
    efficiency or visitation benefit.

    Parameters
    ----------
    SR, seed, budget, M, n_sample_draws
        Same semantics as :func:`run_point`.

    Returns
    -------
    dict
        Flat dict with both arms' observables plus deltas (lr − diag).
        An ``"error"`` field is set on failure; numeric fields are None.
    """
    import blackjax
    from blackjax.adaptation.meta import extract_multi_chain_verdict
    from blackjax.adaptation.staged_adaptation import staged_adaptation
    from blackjax.mcmc.metrics import LowRankInverseMassMatrix

    out: dict = {
        "SR": float(SR),
        "correlated_axes": int(correlated_axes),
        "seed": int(seed),
        "budget": int(budget),
        "M": int(M),
        "init_kind": "broad",
        "n_sample_draws": int(n_sample_draws),
        # Controller verdict from warmup
        "auto_route": None,
        "auto_effective_rank": None,
        "auto_mode_coverage": None,
        "auto_imm_kind": None,
        "metric_route_status": None,
        "metric_route_basis": None,
        "metric_scope": None,
        "observed_ensemble_evidence": None,
        "global_exploration": None,
        "handoff": None,
        "confidence_scope": None,
        "window_events": None,
        "schedule_prescriptions": None,
        # Low-rank arm (deployed IMM)
        "lr_min_ess_per_grad": None,
        "lr_projection_bulk_ess_per_grad": None,
        "lr_projection_tail_ess_per_grad": None,
        "lr_total_integration_steps": None,
        "lr_num_divergences": None,
        "lr_both_modes_frac": None,
        "lr_split_rhat": None,
        "lr_projection_split_rhat": None,
        # Diagonal arm (diagonal of deployed IMM)
        "diag_min_ess_per_grad": None,
        "diag_projection_bulk_ess_per_grad": None,
        "diag_projection_tail_ess_per_grad": None,
        "diag_total_integration_steps": None,
        "diag_num_divergences": None,
        "diag_both_modes_frac": None,
        "diag_split_rhat": None,
        "diag_projection_split_rhat": None,
        # Deltas (lr − diag)
        "delta_min_ess_per_grad": None,
        "projection_bulk_ess_per_grad_ratio": None,
        "delta_both_modes_frac": None,
        "error": None,
    }

    try:
        import arviz as az

        spec = build_gmm(SR, correlated_axes)
        logdensity_fn = spec.logdensity_fn
        mu1_arr = spec.mu1
        mu2_arr = spec.mu2

        positions_np = init_positions(SR, M, seed, "broad", correlated_axes)
        positions = jnp.array(positions_np)

        rng = jax.random.key(seed)
        warmup_key, sample_key_lr, sample_key_diag = jax.random.split(rng, 3)

        # ---- Single auto warmup (M chains) ----
        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=budget,
            n_chains=M,
        )
        results, warmup_info = warmup.run(warmup_key, positions)

        from window_events import extract_window_events

        window_events, schedules = extract_window_events(
            warmup_info,
            max_grad_budget=budget,
            n_chains=M,
            dimension=_D,
        )
        out["window_events"] = window_events
        out["schedule_prescriptions"] = schedules

        final_imm_state = jax.tree_util.tree_map(
            lambda x: x[-1], warmup_info.adaptation_state.imm_state
        )
        num_steps = int(np.asarray(final_imm_state.budget_used))
        verdict = extract_multi_chain_verdict(
            final_imm_state, max_grad_budget=budget, num_warmup_steps=num_steps
        )
        out["auto_route"] = verdict.route
        out["auto_effective_rank"] = int(verdict.effective_rank)
        out["auto_mode_coverage"] = str(verdict.flags.get("mode_coverage", "unknown"))
        out["metric_route_status"] = str(
            verdict.flags.get("metric_route_status", "unassessed")
        )
        out["metric_route_basis"] = str(
            verdict.flags.get("metric_route_basis", "none")
        )
        out["metric_scope"] = str(
            verdict.flags.get("metric_scope", "unassessed")
        )
        out["observed_ensemble_evidence"] = str(
            verdict.flags.get("observed_ensemble_evidence", "unassessed")
        )
        out["global_exploration"] = str(
            verdict.flags.get("global_exploration", "not_established")
        )
        out["handoff"] = str(verdict.flags.get("handoff", "none"))
        out["confidence_scope"] = str(
            verdict.flags.get(
                "confidence_scope",
                "historical_route_selection_heuristic",
            )
        )

        # ---- Extract deployed IMM and its diagonal ----
        step_size = results.parameters["step_size"]
        lr_imm = results.parameters["inverse_mass_matrix"]

        if isinstance(lr_imm, LowRankInverseMassMatrix):
            out["auto_imm_kind"] = "low_rank"
            # M^{-1} = diag(sigma) (I + U(Λ-I)U^T) diag(sigma)
            # diagonal[i] = sigma[i]^2 * (1 + sum_k (lam[k]-1) * U[i,k]^2)
            imm_diag = lr_imm.sigma**2 * (
                1.0 + jnp.sum((lr_imm.lam - 1.0) * lr_imm.U**2, axis=1)
            )
        elif jnp.asarray(lr_imm).ndim == 2:
            out["auto_imm_kind"] = "dense"
            # Dense matrix: extract diagonal
            imm_diag = jnp.diag(jnp.asarray(lr_imm))
        else:
            out["auto_imm_kind"] = "diagonal"
            # Already 1-D (diagonal route): arms are identical, Δ = 0
            imm_diag = jnp.asarray(lr_imm)

        # ---- Shared sampling helper: M chains, vmap + scan ----
        init_states = results.state  # end-of-warmup NUTS states for all M chains

        def _run_sampling(imm, sample_key):
            kernel = blackjax.nuts(
                logdensity_fn, step_size=step_size, inverse_mass_matrix=imm
            )

            def one_step(states, key):
                chain_keys = jax.random.split(key, M)
                new_states, infos = jax.vmap(kernel.step)(chain_keys, states)
                return new_states, (new_states, infos)

            keys = jax.random.split(sample_key, n_sample_draws)
            _, (all_states, all_infos) = jax.lax.scan(one_step, init_states, keys)
            return all_states, all_infos

        def _observables(all_states, all_infos):
            positions_draws = np.asarray(all_states.position)  # (n_draws, M, d)
            d_dim = positions_draws.shape[2]
            draws_per_chain = np.transpose(positions_draws, (1, 0, 2))

            rhat_vals = [
                float(az.rhat({"x": draws_per_chain[:, :, i]})["x"])
                for i in range(d_dim)
            ]
            split_rhat = float(np.nanmax(rhat_vals)) if rhat_vals else float("nan")

            total_grads = int(np.asarray(all_infos.num_integration_steps).sum())
            ess_vals = [
                float(
                    az.ess(
                        {"x": draws_per_chain[:, :, i]}, method="bulk"
                    )["x"]
                )
                for i in range(d_dim)
            ]
            min_ess_raw = float(np.nanmin(ess_vals)) if ess_vals else 0.0
            min_ess = 0.0 if math.isnan(min_ess_raw) else min_ess_raw
            min_ess_per_grad = min_ess / max(total_grads, 1)

            projection_draws = correlated_projection(
                draws_per_chain, correlated_axes
            )
            projection_bulk_ess = float(
                az.ess({"x": projection_draws}, method="bulk")["x"]
            )
            projection_tail_ess = float(
                az.ess({"x": projection_draws}, method="tail")["x"]
            )
            projection_split_rhat = float(
                az.rhat({"x": projection_draws})["x"]
            )

            chain_has_mu1 = np.zeros(M, dtype=bool)
            chain_has_mu2 = np.zeros(M, dtype=bool)
            for c in range(M):
                cd = draws_per_chain[c]
                d1 = np.sum((cd - mu1_arr) ** 2, axis=1)
                d2 = np.sum((cd - mu2_arr) ** 2, axis=1)
                chain_has_mu1[c] = bool(np.any(d1 < d2))
                chain_has_mu2[c] = bool(np.any(d2 < d1))
            both_raw = float(np.mean(chain_has_mu1 & chain_has_mu2))
            both_modes_frac = 0.0 if math.isnan(both_raw) else both_raw
            num_divergences = int(
                np.sum(np.asarray(all_infos.is_divergent))
            )
            return {
                "split_rhat": split_rhat,
                "min_ess_per_grad": min_ess_per_grad,
                "projection_bulk_ess_per_grad": (
                    projection_bulk_ess / max(total_grads, 1)
                ),
                "projection_tail_ess_per_grad": (
                    projection_tail_ess / max(total_grads, 1)
                ),
                "projection_split_rhat": projection_split_rhat,
                "total_integration_steps": total_grads,
                "num_divergences": num_divergences,
                "both_modes_frac": both_modes_frac,
            }

        # Deployed inverse-mass-matrix sampling arm.
        states_lr, infos_lr = _run_sampling(lr_imm, sample_key_lr)
        obs_lr = _observables(states_lr, infos_lr)
        for key, value in obs_lr.items():
            out[f"lr_{key}"] = value

        # Matched-diagonal sampling arm.
        states_diag, infos_diag = _run_sampling(imm_diag, sample_key_diag)
        obs_diag = _observables(states_diag, infos_diag)
        for key, value in obs_diag.items():
            out[f"diag_{key}"] = value

        # Deltas (lr − diag)
        out["delta_min_ess_per_grad"] = (
            obs_lr["min_ess_per_grad"] - obs_diag["min_ess_per_grad"]
        )
        out["projection_bulk_ess_per_grad_ratio"] = (
            obs_lr["projection_bulk_ess_per_grad"]
            / obs_diag["projection_bulk_ess_per_grad"]
        )
        out["delta_both_modes_frac"] = (
            obs_lr["both_modes_frac"] - obs_diag["both_modes_frac"]
        )

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return out


# ---------------------------------------------------------------------------
# 6. run_single_chain — single-chain path contrast
# ---------------------------------------------------------------------------


def run_single_chain(
    SR: float,
    seed: int,
    budget: int,
    n_sample_draws: int = 200,
    correlated_axes: int = 2,
) -> dict:
    """Run the single-chain meta controller (n_chains=1) for contrast with multi-chain.

    Uses :func:`~blackjax.adaptation.meta.extract_meta_verdict` (single-chain
    extractor, NOT the multi-chain extractor).  Expected behaviour: the single-
    chain path never sets ``deferred_to_ensemble`` (field absent) and always
    reports ``mode_coverage='single_chain_uncertified'`` — it cannot certify
    multi-modal coverage.

    Parameters
    ----------
    SR, seed, budget, n_sample_draws
        Same semantics as :func:`run_point`.

    Returns
    -------
    dict
        Single-chain verdict fields + sampling observables.
    """
    import blackjax
    from blackjax.adaptation.meta import extract_meta_verdict
    from blackjax.adaptation.staged_adaptation import staged_adaptation

    out: dict = {
        "SR": float(SR),
        "correlated_axes": int(correlated_axes),
        "seed": int(seed),
        "budget": int(budget),
        "n_chains": 1,
        "init_kind": "broad",
        "n_sample_draws": int(n_sample_draws),
        "route": None,
        "effective_rank": None,
        "confidence": None,
        "mode_coverage": None,
        "metric_route_status": None,
        "metric_route_basis": None,
        "metric_scope": None,
        "observed_ensemble_evidence": None,
        "global_exploration": "not_established",
        "handoff": None,
        "confidence_scope": None,
        "window_events": None,
        "schedule_prescriptions": None,
        # Single-chain path has no deferred_to_ensemble or unimodality_gate.
        "deferred_to_ensemble": None,  # not produced by single-chain path
        "unimodality_gate": None,  # not produced by single-chain path
        "split_rhat": None,
        "bulk_ess_per_grad": None,
        "mode_weight_est": None,
        "both_modes_visited": None,
        "num_divergences": None,
        "error": None,
    }

    try:
        import arviz as az

        spec = build_gmm(SR, correlated_axes)
        logdensity_fn = spec.logdensity_fn

        # Single-chain init: broad draw from marginal.
        positions_np = init_positions(SR, 1, seed, "broad", correlated_axes)
        position = jnp.array(positions_np[0])  # (d,)

        rng = jax.random.key(seed)
        warmup_key, sample_key = jax.random.split(rng)

        # Run single-chain warmup (n_chains=1 is the default).
        warmup = staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=budget,
            n_chains=1,
        )
        results, warmup_info = warmup.run(warmup_key, position)

        from window_events import extract_window_events

        window_events, schedules = extract_window_events(
            warmup_info,
            max_grad_budget=budget,
            n_chains=1,
            dimension=_D,
        )
        out["window_events"] = window_events
        out["schedule_prescriptions"] = schedules

        # Extract single-chain MetaAdaptationCoreState (NOT MultiChain variant).
        final_imm_state = jax.tree_util.tree_map(
            lambda x: x[-1], warmup_info.adaptation_state.imm_state
        )
        num_warmup_steps = int(np.asarray(final_imm_state.budget_used))

        # Single-chain verdict extractor.
        verdict = extract_meta_verdict(
            final_imm_state,
            max_grad_budget=budget,
            num_warmup_steps=num_warmup_steps,
        )

        out["route"] = verdict.route
        out["effective_rank"] = int(verdict.effective_rank)
        out["confidence"] = verdict.confidence
        out["mode_coverage"] = str(
            verdict.flags.get("mode_coverage", "single_chain_uncertified")
        )
        out["metric_route_status"] = str(
            verdict.flags.get("metric_route_status", "unassessed")
        )
        out["metric_route_basis"] = str(
            verdict.flags.get("metric_route_basis", "single_chain_spectrum")
        )
        out["metric_scope"] = str(
            verdict.flags.get("metric_scope", "within_chain_conditional")
        )
        out["observed_ensemble_evidence"] = str(
            verdict.flags.get("observed_ensemble_evidence", "unassessed")
        )
        out["global_exploration"] = str(
            verdict.flags.get("global_exploration", "not_established")
        )
        out["handoff"] = str(verdict.flags.get("handoff", "none"))
        out["confidence_scope"] = str(
            verdict.flags.get(
                "confidence_scope",
                "historical_route_selection_heuristic",
            )
        )
        # These fields are genuinely absent in the single-chain path.
        out["deferred_to_ensemble"] = None
        out["unimodality_gate"] = None

        # Post-warmup single-chain NUTS sampling.
        step_size = results.parameters["step_size"]
        imm = results.parameters["inverse_mass_matrix"]
        nuts = blackjax.nuts(
            logdensity_fn, step_size=step_size, inverse_mass_matrix=imm
        )

        def one_step(state, key):
            new_state, info = nuts.step(key, state)
            return new_state, (new_state, info)

        sample_keys = jax.random.split(sample_key, n_sample_draws)
        _, (all_states, all_infos) = jax.lax.scan(one_step, results.state, sample_keys)

        # all_states.position shape: (n_draws, d)
        positions_draws = np.asarray(all_states.position)  # (n_draws, d)
        d = positions_draws.shape[1]
        draws_1chain = positions_draws[np.newaxis, :, :]  # (1, n_draws, d)

        # Split-R-hat is undefined for one chain; record a JSON null rather
        # than serializing a NaN that can be mistaken for a computed result.
        out["split_rhat"] = None
        ess_values = [
            float(az.ess({"x": draws_1chain[:, :, i]}, method="bulk")["x"])
            for i in range(d)
        ]
        total_grads = int(np.asarray(all_infos.num_integration_steps).sum())
        out["bulk_ess_per_grad"] = min(ess_values) / max(total_grads, 1)

        flat = positions_draws
        dist1 = np.sum((flat - spec.mu1) ** 2, axis=1)
        dist2 = np.sum((flat - spec.mu2) ** 2, axis=1)
        near_mu2 = dist2 < dist1
        out["mode_weight_est"] = float(np.mean(near_mu2))
        out["both_modes_visited"] = bool(np.any(near_mu2) and np.any(~near_mu2))
        out["num_divergences"] = int(
            np.sum(np.asarray(all_infos.is_divergent))
        )

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return out


# ---------------------------------------------------------------------------
# 7. Figures (real implementations replacing stubs)
# ---------------------------------------------------------------------------


def _load_jsonl(jsonl_path: str | Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    rows = []
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl_row(fh: TextIO, row: dict) -> None:
    """Append one completed result row and make it visible to recovery readers."""
    def _clean(value: Any) -> Any:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, np.ndarray):
            return [_clean(item) for item in value.tolist()]
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(item) for item in value]
        return value

    fh.write(json.dumps(_clean(row), allow_nan=False, sort_keys=True) + "\n")
    fh.flush()


def _open_exclusive_jsonl(path: str | Path) -> TextIO:
    """Create a new JSONL file without replacing an existing raw result."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("x", encoding="utf-8")


def _median_by_sr(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    """Return (sr_sorted, median_values) for a numeric field, grouped by SR."""
    from collections import defaultdict

    by_sr: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("error") is None and r.get(key) is not None:
            v = r[key]
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            ):
                by_sr[float(r["SR"])].append(float(v))
    srs_sorted = sorted(by_sr)
    medians = [float(np.median(by_sr[s])) for s in srs_sorted]
    return srs_sorted, medians


def plot_money_panel(
    results: list[dict], out_path: str, correlated_axes: int = 2
) -> None:
    """Money panel: true-GMM scatter strip + controller signal curves vs SR.

    Layout
    ------
    Top strip
        One 2D scatter inset per selected SR value (SR ∈ {1.5, 4, 6, 8, 10}, or
        nearest available).  Samples drawn DIRECTLY from the true GMM
        (2000 draws per SR: pick component by weight, draw from N(μ_k, Σ_within)).
        Shows (x0, x1) — the separation is along v = (1,1,0,0,0)/√2 so the two
        clusters move apart diagonally in this view.  μ₁ (×) and μ₂ (+) overlaid.

    Bottom
        Left y-axis: within_λ₁ (median) + flat marginal λ₁=1.913 reference line.
        Right y-axis: chain_consistency_psi (Ψ) and chain_collinearity.
        Single contiguous shading from the FIRST SR where warning fires to the
        end of the sweep (one region, one label "warning fired").
        Dotted connection lines link each scatter inset to its SR on the x-axis.

    Accepts a list of result dicts (as returned by run_point or loaded from JSONL).
    """
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch

        marginal_spike = analytic_diagnostic(correlated_axes)[
            "marginal_whitened_spike"
        ]

        valid = [r for r in results if r.get("error") is None]
        if not valid:
            print("[plot_money_panel] no valid rows", file=sys.stderr)
            return

        srs_lam, med_lam = _median_by_sr(valid, "within_lam1")
        srs_psi, med_psi = _median_by_sr(valid, "chain_consistency_psi")
        srs_col, med_col = _median_by_sr(valid, "chain_collinearity")

        all_srs = sorted({float(r["SR"]) for r in valid})

        # First deferred SR → one contiguous warning region to end of sweep.
        deferred_srs = sorted(
            {float(r["SR"]) for r in valid if r.get("deferred_to_ensemble") is True}
        )

        # Transition line.
        low_rank_srs = {float(r["SR"]) for r in valid if r.get("route") == "low_rank"}
        diag_srs_set = {float(r["SR"]) for r in valid if r.get("route") == "diagonal"}
        transition_sr = None
        if low_rank_srs and diag_srs_set:
            transition_sr = min(diag_srs_set - low_rank_srs, default=None)

        # Strip insets: find the available SR closest to each target.
        strip_targets = [1.5, 4.0, 6.0, 8.0, 10.0]
        strip_srs: list[float] = []
        for t in strip_targets:
            if all_srs:
                closest = min(all_srs, key=lambda x: abs(x - t))
                if closest not in strip_srs:
                    strip_srs.append(closest)
        n_strip = len(strip_srs)

        # --- Figure layout: top strip + bottom signal panel ---
        fig = plt.figure(figsize=(11, 6.5))
        gs = gridspec.GridSpec(
            2,
            n_strip,
            figure=fig,
            height_ratios=[1.0, 2.5],
            hspace=0.50,
            wspace=0.38,
        )
        ax_main = fig.add_subplot(gs[1, :])  # main curves span all columns
        inset_ax_list = [fig.add_subplot(gs[0, i]) for i in range(n_strip)]

        # --- True-GMM scatter insets ---
        for ax_in, sr in zip(inset_ax_list, strip_srs):
            spec_s = build_gmm(sr, correlated_axes)
            rng_np = np.random.default_rng(int(sr * 100 + 42))
            n_true = 2000
            comp = rng_np.choice(2, size=n_true, p=[_PI1, _PI2])
            L_w = np.linalg.cholesky(spec_s.Sigma_within)
            noise = rng_np.standard_normal((n_true, _D)) @ L_w.T
            # Component samples: mu1 + noise  or  mu2 + noise
            samp1 = spec_s.mu1 + noise[comp == 0]
            samp2 = spec_s.mu2 + noise[comp == 1]
            ax_in.scatter(
                samp1[:, 0],
                samp1[:, 1],
                s=1.5,
                alpha=0.25,
                color="steelblue",
                rasterized=True,
            )
            ax_in.scatter(
                samp2[:, 0],
                samp2[:, 1],
                s=1.5,
                alpha=0.25,
                color="darkorange",
                rasterized=True,
            )
            ax_in.plot(
                spec_s.mu1[0],
                spec_s.mu1[1],
                "x",
                color="steelblue",
                ms=6,
                mew=1.8,
                zorder=5,
            )
            ax_in.plot(
                spec_s.mu2[0],
                spec_s.mu2[1],
                "+",
                color="darkorange",
                ms=6,
                mew=1.8,
                zorder=5,
            )
            ax_in.set_title(f"SR={sr:.1f}", fontsize=7, pad=2)
            ax_in.set_xlabel("x₀", fontsize=6, labelpad=1)
            ax_in.set_ylabel("x₁", fontsize=6, labelpad=1)
            ax_in.tick_params(labelsize=5)

        # --- Main signal curves ---
        ax1 = ax_main
        ax1.plot(srs_lam, med_lam, "o-", color="steelblue", label="within λ₁ (median)")
        ax1.axhline(
            marginal_spike,
            ls="--",
            color="steelblue",
            alpha=0.5,
            label=f"marginal λ₁ = {marginal_spike:.3f}",
        )
        ax1.set_xlabel("SR (separation ratio)")
        ax1.set_ylabel("within_lam1", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_ylim(bottom=0.9)
        if all_srs:
            ax1.set_xlim(min(all_srs) - 0.3, max(all_srs) + 0.3)

        ax2 = ax1.twinx()
        if srs_psi:
            ax2.plot(
                srs_psi,
                med_psi,
                "s--",
                color="darkorange",
                alpha=0.8,
                label="chain_consistency_psi (Ψ)",
            )
        if srs_col:
            ax2.plot(
                srs_col,
                med_col,
                "^:",
                color="green",
                alpha=0.8,
                label="chain_collinearity",
            )
        ax2.set_ylabel("Ψ / collinearity", color="grey")
        ax2.set_ylim(-0.1, 1.1)

        # Single contiguous warning region from first deferred SR to sweep end.
        if deferred_srs and all_srs:
            ax1.axvspan(
                deferred_srs[0],
                max(all_srs) + 0.3,
                alpha=0.12,
                color="red",
                label="warning fired",
            )

        # Transition line.
        if transition_sr is not None:
            ax1.axvline(
                transition_sr,
                ls="-.",
                color="purple",
                alpha=0.7,
                label=f"low_rank→diagonal @ SR={transition_sr}",
            )

        # Combined legend.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        ax1.set_title("Controller signals vs separation ratio SR")

        # Dotted connection lines from bottom-center of each inset to its SR on ax1.
        xlim = ax1.get_xlim()
        for ax_in, sr in zip(inset_ax_list, strip_srs):
            try:
                x_norm = (sr - xlim[0]) / (xlim[1] - xlim[0])
                con = ConnectionPatch(
                    xyA=(0.5, 0.0),  # bottom-center of inset (axes fraction)
                    xyB=(x_norm, 1.0),  # top of main axes at SR x (axes fraction)
                    coordsA="axes fraction",
                    coordsB="axes fraction",
                    axesA=ax_in,
                    axesB=ax1,
                    color="grey",
                    linestyle=":",
                    lw=0.8,
                    alpha=0.55,
                )
                fig.add_artist(con)
            except Exception:
                pass

        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_money_panel] saved {out_path}")
    except Exception as exc:
        print(f"[plot_money_panel] failed: {exc}", file=sys.stderr)


def plot_imm_2d(results: list[dict], out_path: str, correlated_axes: int = 2) -> None:
    """Deployed vs analytic IMM (x1,x2) 2×2 block across SR.

    For each SR in the results, computes:

    * ``Σ_within⁻¹`` (x1,x2) block: the block the controller should deploy
      if it identifies the within-mode structure perfectly.
    * ``Σ_marginal⁻¹`` (x1,x2) block: the optimal block for the *mixed*
      marginal (constant reference).

    Plots the (0,0) diagonal element and (0,1) off-diagonal element of each
    block, overlaying the controller's effective_rank to show when escalation
    occurred.

    Note: the DEPLOYED IMM is not stored in JSONL output (only route/rank/flags
    are recorded).  This figure shows the ANALYTIC blocks that the ideal
    controller would discover, letting the viewer relate ``effective_rank`` and
    ``within_lam1`` to the underlying geometry.
    """
    try:
        import matplotlib.pyplot as plt

        valid = [r for r in results if r.get("error") is None]
        if not valid:
            print("[plot_imm_2d] no valid rows", file=sys.stderr)
            return

        # Analytic blocks from GMMSpec.
        sr_list = sorted({float(r["SR"]) for r in valid})
        imm_within_diag = []
        imm_within_offdiag = []
        imm_marg_diag = []
        imm_marg_offdiag = []

        for sr in sr_list:
            spec = build_gmm(sr, correlated_axes)
            sw_inv = np.linalg.inv(spec.Sigma_within)
            sm_inv = np.linalg.inv(spec.Sigma_marginal)
            imm_within_diag.append(float(sw_inv[0, 0]))
            imm_within_offdiag.append(float(sw_inv[0, 1]))
            imm_marg_diag.append(float(sm_inv[0, 0]))
            imm_marg_offdiag.append(float(sm_inv[0, 1]))

        # Median effective_rank from results by SR.
        srs_rk, med_rk = _median_by_sr(valid, "effective_rank")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Panel 1: diagonal element (0,0).
        axes[0].plot(
            sr_list,
            imm_within_diag,
            "o-",
            color="steelblue",
            label="Σ_within⁻¹[0,0] (analytic)",
        )
        axes[0].axhline(
            imm_marg_diag[0],
            ls="--",
            color="steelblue",
            alpha=0.5,
            label="Σ_marginal⁻¹[0,0] (const)",
        )
        ax2_0 = axes[0].twinx()
        ax2_0.plot(
            srs_rk,
            med_rk,
            "s:",
            color="darkorange",
            alpha=0.7,
            label="effective_rank (median)",
        )
        ax2_0.set_ylabel("effective_rank", color="darkorange")
        ax2_0.tick_params(axis="y", labelcolor="darkorange")
        axes[0].set_xlabel("SR")
        axes[0].set_ylabel("IMM diagonal [0,0]")
        axes[0].set_title("IMM (x1,x2) diagonal vs SR")
        axes[0].legend(fontsize=7)

        # Panel 2: off-diagonal element (0,1).
        axes[1].plot(
            sr_list,
            imm_within_offdiag,
            "o-",
            color="darkorange",
            label="Σ_within⁻¹[0,1] (analytic)",
        )
        axes[1].axhline(
            imm_marg_offdiag[0],
            ls="--",
            color="darkorange",
            alpha=0.5,
            label="Σ_marginal⁻¹[0,1] (const)",
        )
        axes[1].set_xlabel("SR")
        axes[1].set_ylabel("IMM off-diagonal [0,1]")
        axes[1].set_title("IMM (x1,x2) off-diagonal vs SR")
        axes[1].legend(fontsize=7)

        fig.suptitle(
            "Analytic IMM (x1,x2) 2×2 block vs SR\n"
            "(deployed IMM not stored in JSONL; blocks shown are analytic targets)"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"[plot_imm_2d] saved {out_path}")
    except Exception as exc:
        print(f"[plot_imm_2d] failed: {exc}", file=sys.stderr)


def plot_data_scatter(
    results: list[dict], out_path: str, correlated_axes: int = 2
) -> None:
    """Sampling quality scatter: split_rhat, mode_weight_est, both_modes_frac vs SR.

    Shows three panels:
    1. split_rhat vs SR — reference line at 1.01.
    2. mode_weight_est vs SR — reference line at true π₂=0.70.
    3. both_modes_visited_frac vs SR — per-point scatter, plus the analytic
       μ₁ and μ₂ positions annotated in a text box.

    Accepts a list of result dicts (as returned by run_point or loaded from JSONL).
    Each dot is one (SR, seed) point; the dashed line is the median per SR.
    """
    try:
        import matplotlib.pyplot as plt

        valid = [r for r in results if r.get("error") is None]
        if not valid:
            print("[plot_data_scatter] no valid rows", file=sys.stderr)
            return

        srs_r, med_rhat = _median_by_sr(valid, "split_rhat")
        srs_m, med_mw = _median_by_sr(valid, "mode_weight_est")
        srs_b, med_both = _median_by_sr(valid, "both_modes_visited_frac")

        # Raw scatter points.
        def _raw(key):
            pts = [
                (float(r["SR"]), float(r[key]))
                for r in valid
                if r.get(key) is not None
                and isinstance(r[key], (int, float))
                and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return zip(*pts) if pts else ([], [])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Panel 1: split_rhat.
        xs_r, ys_r = _raw("split_rhat")
        axes[0].scatter(xs_r, ys_r, s=20, alpha=0.5, color="steelblue")
        if srs_r:
            axes[0].plot(srs_r, med_rhat, "k-", lw=1.5, label="median")
        axes[0].axhline(1.01, ls="--", color="red", alpha=0.7, label="R̂=1.01")
        axes[0].set_xlabel("SR")
        axes[0].set_ylabel("split_rhat (max over dims)")
        axes[0].set_title("split R̂ vs SR")
        axes[0].legend(fontsize=8)

        # Panel 2: mode_weight_est.
        xs_m, ys_m = _raw("mode_weight_est")
        axes[1].scatter(xs_m, ys_m, s=20, alpha=0.5, color="darkorange")
        if srs_m:
            axes[1].plot(srs_m, med_mw, "k-", lw=1.5, label="median")
        axes[1].axhline(0.70, ls="--", color="green", alpha=0.7, label="true π₂=0.70")
        axes[1].set_xlabel("SR")
        axes[1].set_ylabel("mode_weight_est (P(nearest μ₂))")
        axes[1].set_title("Mode weight estimate vs SR")
        axes[1].legend(fontsize=8)

        # Panel 3: both_modes_visited_frac.
        xs_b, ys_b = _raw("both_modes_visited_frac")
        axes[2].scatter(xs_b, ys_b, s=20, alpha=0.5, color="green")
        if srs_b:
            axes[2].plot(srs_b, med_both, "k-", lw=1.5, label="median")
        # Annotate μ1 and μ2 positions along v (first component) for reference.
        sr_examples = [1.5, 4.0, 7.0, 9.0]
        mu_info = []
        for sr in sr_examples:
            spec = build_gmm(sr, correlated_axes)
            mu1_x1 = float(spec.mu1[0])
            mu2_x1 = float(spec.mu2[0])
            mu_info.append(f"SR={sr}: μ₁[0]={mu1_x1:.1f}, μ₂[0]={mu2_x1:.1f}")
        axes[2].text(
            0.02,
            0.98,
            "\n".join(mu_info),
            transform=axes[2].transAxes,
            fontsize=6,
            va="top",
            family="monospace",
        )
        axes[2].set_xlabel("SR")
        axes[2].set_ylabel("both_modes_visited_frac")
        axes[2].set_title("Cross-mode mixing fraction vs SR")
        axes[2].legend(fontsize=8)

        fig.suptitle(
            "Sampling quality vs separation ratio SR (multi-chain, broad init)"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"[plot_data_scatter] saved {out_path}")
    except Exception as exc:
        print(f"[plot_data_scatter] failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------

_SMOKE_SR = [1.5, 9.0]
_SWEEP_SR = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0]
_SINGLE_CHAIN_SR = [1.5, 5.0, 10.0]
_SMOKE_SEEDS = [42, 123]
_SWEEP_SEEDS = [42, 123, 456]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5D GMM boundary experiment for the multi-chain meta controller."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke", action="store_true", help="Run 2-3 SR points (quick check)"
    )
    mode.add_argument("--sweep", action="store_true", help="Full SR grid")
    mode.add_argument(
        "--ablation",
        action="store_true",
        help="Run auto vs diagonal ablation at a few SR values",
    )
    mode.add_argument(
        "--single-chain",
        action="store_true",
        dest="single_chain",
        help="Run single-chain (n_chains=1) controller for contrast",
    )
    mode.add_argument(
        "--figures",
        metavar="JSONL",
        default=None,
        help="Generate all three figures from the given JSONL file",
    )
    mode.add_argument(
        "--diagnose-historical-label",
        action="store_true",
        dest="diagnose_historical_label",
        help="Re-run SR=10/seed=42 and dump the corrected verdict fields",
    )
    mode.add_argument(
        "--analytic",
        action="store_true",
        help="Report analytic geometry and invariance without running a sampler",
    )
    parser.add_argument("--budget", type=int, default=60_000, help="max_grad_budget")
    parser.add_argument(
        "--init", default="broad", choices=["broad", "mode_centered", "split"]
    )
    parser.add_argument(
        "--out",
        default=None,
        help="New incremental output JSONL path (required for sampling modes)",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=None,
        help="Post-warmup draws per chain (defaults: 200 single-chain, 1000 otherwise)",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp",
        help="Output directory for figures (default: /tmp)",
    )
    parser.add_argument(
        "--correlated-axes",
        type=int,
        default=2,
        metavar="K",
        help="Number of leading axes in the correlated direction (2 through 5)",
    )
    args = parser.parse_args()
    correlated_axes = args.correlated_axes
    _correlated_direction(correlated_axes)
    sampling_mode = args.smoke or args.sweep or args.ablation or args.single_chain
    if sampling_mode and args.out is None:
        parser.error("--out is required for every sampling mode")
    if args.n_draws is not None and args.n_draws < 4:
        parser.error("--n-draws must be at least 4")

    if args.analytic:
        diagnostic = analytic_diagnostic(correlated_axes)
        max_err, lam1_marg, lam1_within = verify_invariance(
            list(np.linspace(0, 10, 21)), correlated_axes
        )
        print(json.dumps(diagnostic, sort_keys=True))
        print(f"max_inf_norm_err = {max_err:.2e}")
        print(f"lam1_marginal = {lam1_marg:.6f}")
        print(
            "lam1_within at SR=0,5,10 = "
            f"{lam1_within[0]:.3f}, {lam1_within[10]:.3f}, {lam1_within[20]:.3f}"
        )
        return

    # ---- figures mode: no heavy compute ----
    if args.figures:
        rows = _load_jsonl(args.figures)
        print(f"Loaded {len(rows)} rows from {args.figures}")
        import os

        out_dir = args.out_dir
        plot_money_panel(rows, os.path.join(out_dir, "gmm_money_panel.png"), correlated_axes)
        plot_imm_2d(rows, os.path.join(out_dir, "gmm_imm_2d.png"), correlated_axes)
        plot_data_scatter(rows, os.path.join(out_dir, "gmm_data_scatter.png"), correlated_axes)
        print(f"Figures written to {out_dir}/gmm_*.png")
        return

    # ---- Historical-label diagnostic ----
    if args.diagnose_historical_label:
        import blackjax
        from blackjax.adaptation.meta import extract_multi_chain_verdict
        from blackjax.adaptation.staged_adaptation import staged_adaptation

        diagnostic_sr, diagnostic_seed = 10.0, 42
        spec = build_gmm(diagnostic_sr, correlated_axes)
        positions = jnp.array(
            init_positions(
                diagnostic_sr,
                8,
                diagnostic_seed,
                "broad",
                correlated_axes,
            )
        )
        rng = jax.random.key(diagnostic_seed)
        warmup_key, _ = jax.random.split(rng)
        warmup = staged_adaptation(
            blackjax.nuts,
            spec.logdensity_fn,
            metric="auto",
            max_grad_budget=args.budget,
            n_chains=8,
        )
        results, warmup_info = warmup.run(warmup_key, positions)
        final_imm = jax.tree_util.tree_map(
            lambda x: x[-1], warmup_info.adaptation_state.imm_state
        )
        num_steps = int(np.asarray(final_imm.budget_used))
        verdict = extract_multi_chain_verdict(
            final_imm, max_grad_budget=args.budget, num_warmup_steps=num_steps
        )
        print(
            "=== corrected verdict "
            f"(SR={diagnostic_sr}, seed={diagnostic_seed}) ==="
        )
        print(f"route: {verdict.route}")
        print(f"effective_rank: {verdict.effective_rank}")
        print(f"confidence: {verdict.confidence}")
        print("flags:")
        for k, v in sorted(verdict.flags.items()):
            print(f"  {k}: {v!r}")
        print("\ncore state (selected):")
        for field in (
            "has_escalated",
            "deferred_to_ensemble",
            "unimodality_passed",
            "unimodality_flag_count",
            "detection_branch",
            "chain_collinearity",
            "chain_consistency_psi",
            "within_lam1",
            "r1_top",
        ):
            if hasattr(final_imm, field):
                v = getattr(final_imm, field)
                try:
                    print(f"  {field}: {float(np.asarray(v)):.6g}")
                except Exception:
                    print(f"  {field}: {v!r}")
        # Interpretation
        print(
            "\nInterpretation: the historical frozen run exposed a positive coverage-label "
            "defect.  In the current verdict, metric route status and basis, metric scope, "
            "observed-ensemble evidence, global_exploration, handoff, and confidence scope "
            "are separate fields; global exploration is not established.  Compare new "
            "output only after the post-repair sweep."
        )
        return

    # ---- single-chain mode ----
    if args.single_chain:
        print("=== single-chain controller contrast ===")
        draws = args.n_draws if args.n_draws is not None else 200
        with _open_exclusive_jsonl(args.out) as output:
            for sr in _SINGLE_CHAIN_SR:
                t0 = time.time()
                rec = run_single_chain(
                    sr,
                    seed=42,
                    budget=args.budget,
                    n_sample_draws=draws,
                    correlated_axes=correlated_axes,
                )
                rec["elapsed_seconds"] = time.time() - t0
                _append_jsonl_row(output, rec)
                print(
                    f"\nSR={sr:.1f}  route={rec['route']}  "
                    f"eff_rank={rec['effective_rank']}"
                )
                print(
                    f"  mode_coverage={rec['mode_coverage']}  "
                    f"confidence={rec['confidence']}"
                )
                print(
                    f"  global_exploration={rec['global_exploration']}  "
                    f"mode_weight_est={rec['mode_weight_est']}"
                )
                if rec["error"]:
                    print(f"  ERROR: {rec['error'][:300]}")
                print(f"  elapsed: {rec['elapsed_seconds']:.1f}s")
        print("\n=== done ===")
        return

    # ---- ablation mode ----
    if args.ablation:
        print("=== ablation: low-rank (deployed) vs diagonal (same-warmup) ===")
        ablation_srs = [1.5, 4.0, 5.0, 9.0]
        draws = args.n_draws if args.n_draws is not None else 1000
        with _open_exclusive_jsonl(args.out) as output:
            for sr in ablation_srs:
                t0 = time.time()
                rec = run_ablation(
                    sr,
                    seed=42,
                    budget=args.budget,
                    n_sample_draws=draws,
                    correlated_axes=correlated_axes,
                )
                rec["elapsed_seconds"] = time.time() - t0
                _append_jsonl_row(output, rec)
                print(f"\nSR={sr:.1f}")
                print(
                    f"  warmup: route={rec['auto_route']} "
                    f"eff_rank={rec['auto_effective_rank']} "
                    f"global_exploration={rec['global_exploration']}"
                )
                if rec["error"]:
                    print(f"  ERROR: {rec['error'][:300]}")
                else:
                    print(
                        f"  lr: min_ess/grad={rec['lr_min_ess_per_grad']:.5f}  "
                        f"both_modes={rec['lr_both_modes_frac']:.2f}  "
                        f"rhat={rec['lr_split_rhat']:.3f}"
                    )
                    print(
                        f"  diag: min_ess/grad={rec['diag_min_ess_per_grad']:.5f}  "
                        f"both_modes={rec['diag_both_modes_frac']:.2f}  "
                        f"rhat={rec['diag_split_rhat']:.3f}"
                    )
                    print(
                        f"  delta_ess={rec['delta_min_ess_per_grad']:.5f}  "
                        f"delta_modes={rec['delta_both_modes_frac']:.2f}"
                    )
                print(f"  elapsed: {rec['elapsed_seconds']:.1f}s")
        print("\n=== done ===")
        return

    # ---- smoke / sweep ----
    if args.smoke:
        sr_list = _SMOKE_SR
        seeds = _SMOKE_SEEDS[:1]  # single seed for speed
    else:
        sr_list = _SWEEP_SR
        seeds = _SWEEP_SEEDS

    # Always run verify_invariance first.
    print("=== verify_invariance ===")
    max_err, lam1_marg, lam1_within_list = verify_invariance(
        list(np.linspace(0, 10, 21)), correlated_axes
    )
    print(f"  max_inf_norm_err = {max_err:.2e}")
    print(f"  lam1_marginal (constant) = {lam1_marg:.6f}")
    print(
        f"  lam1_within at SR=0,5,10 = {lam1_within_list[0]:.3f}, "
        f"{lam1_within_list[10]:.3f}, {lam1_within_list[20]:.3f}"
    )
    assert max_err < 1e-12, f"Invariance failed: {max_err:.2e} >= 1e-12"
    expected_spike = analytic_diagnostic(correlated_axes)["marginal_whitened_spike"]
    assert abs(lam1_marg - expected_spike) < 1e-12, (
        f"lam1_marginal = {lam1_marg} ≠ {expected_spike}"
    )
    print(f"  [PASS] invariance < 1e-12, lam1_marginal ≈ {expected_spike:.6f}")

    smoke_low_result: dict | None = None
    smoke_high_result: dict | None = None
    persisted_rows = 0
    out_fh: TextIO | None = None
    if args.out:
        out_fh = _open_exclusive_jsonl(args.out)
        print(f"Initialized new output JSONL: {args.out} (persisted_rows=0)")

    try:
        for seed in seeds:
            for sr in sr_list:
                t0 = time.time()
                print(f"\n--- SR={sr:.1f} seed={seed} init={args.init} ---")
                rec = run_point(
                    sr,
                    seed,
                    args.init,
                    args.budget,
                    n_sample_draws=(
                        args.n_draws if args.n_draws is not None else 1000
                    ),
                    correlated_axes=correlated_axes,
                )
                elapsed = time.time() - t0
                if args.smoke and abs(rec["SR"] - 1.5) < 0.01:
                    smoke_low_result = rec
                if args.smoke and abs(rec["SR"] - 9.0) < 0.01:
                    smoke_high_result = rec
                if out_fh is not None:
                    _append_jsonl_row(out_fh, rec)
                    persisted_rows += 1
                    print(f"  persisted_rows={persisted_rows} path={args.out}")

                print(f"  route={rec['route']}  eff_rank={rec['effective_rank']}")
                print(
                    f"  deferred={rec['deferred_to_ensemble']}  "
                    f"detection_branch={rec['detection_branch']}"
                )
                print(
                    f"  within_lam1={rec['within_lam1']}  "
                    f"collinearity={rec['chain_collinearity']:.4f}"
                    if rec["chain_collinearity"] is not None
                    else f"  within_lam1={rec['within_lam1']}  collinearity=None"
                )
                print(
                    f"  unimodality_gate={rec['unimodality_gate']}  "
                    f"mode_coverage={rec['mode_coverage']}"
                )
                print(
                    f"  split_rhat={rec['split_rhat']}  "
                    f"mode_weight_est={rec['mode_weight_est']}"
                )
                print(
                    f"  min_ess_per_grad={rec['min_ess_per_grad']}  "
                    f"both_modes_frac={rec['both_modes_visited_frac']}"
                )
                if rec["error"]:
                    print(f"  ERROR: {rec['error'][:200]}")
                print(f"  elapsed: {elapsed:.1f}s")
    finally:
        if out_fh is not None:
            out_fh.close()

    if args.smoke:
        rec_low = smoke_low_result
        rec_high = smoke_high_result

        if rec_low and not rec_low.get("error"):
            escalation_signal = (
                rec_low["route"] == "low_rank"
                or (
                    rec_low["effective_rank"] is not None
                    and rec_low["effective_rank"] >= 1
                )
                or (rec_low["detection_branch"] not in (None, "none"))
            )
            print(
                f"\nSR=1.5 escalation_signal={escalation_signal} "
                f"(route={rec_low['route']}, eff_rank={rec_low['effective_rank']}, "
                f"branch={rec_low['detection_branch']})"
            )
            assert escalation_signal, (
                f"FAIL: SR=1.5 shows no escalation signal. "
                f"route={rec_low['route']}, eff_rank={rec_low['effective_rank']}, "
                f"branch={rec_low['detection_branch']}"
            )
            print("  [PASS] SR=1.5 escalation signal")
        else:
            print(f"SR=1.5 result missing or errored: {rec_low}")

        if rec_high and not rec_high.get("error"):
            mode_split_signal = rec_high["deferred_to_ensemble"] is True or (
                rec_high["route"] == "diagonal"
                and rec_high["unimodality_gate"] == "flag"
            )
            print(
                f"\nSR=9 mode_split_signal={mode_split_signal} "
                f"(deferred={rec_high['deferred_to_ensemble']}, "
                f"route={rec_high['route']}, "
                f"unimodality={rec_high['unimodality_gate']})"
            )
            assert mode_split_signal, (
                f"FAIL: SR=9 shows no mode-split/refusal signal. "
                f"deferred={rec_high['deferred_to_ensemble']}, "
                f"route={rec_high['route']}, "
                f"unimodality={rec_high['unimodality_gate']}"
            )
            print("  [PASS] SR=9 mode-split/refusal signal")
        else:
            print(f"SR=9 result missing or errored: {rec_high}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
