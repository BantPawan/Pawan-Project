"""Bounded-memory GPU pairwise Pearson correlations.

The GPU computes the pairwise-complete sufficient statistics in float64:
N=M.T@M, P=X.T@X, S=X.T@M, Q=(X*X).T@M. X is centered and
scaled per column, with missing values zero-filled and M its finite mask.
Rows are streamed once to the GPU. Column blocks limit GEMM temporaries.
Targeted pandas recomputation protects ill-conditioned pairwise subsets
and decisions very close to a supplied feature-selection cutoff.

The float64 calculation preserves pairwise missing-value semantics, with
targeted pandas recomputation for ill-conditioned and cutoff-adjacent pairs.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import pandas as pd


def _column_parameters(frame):
    """Compute stable translation/scaling parameters using one column at a time."""
    size = frame.shape[1]
    anchors = np.zeros(size)
    scales = np.ones(size)
    pre_scales = np.ones(size)
    means = np.zeros(size)
    active = np.zeros(size, dtype=bool)
    extreme = np.zeros(size, dtype=bool)
    for index in range(size):
        raw = frame.iloc[:, index].to_numpy(dtype=np.float64, na_value=np.nan)
        values = raw[np.isfinite(raw)]
        if len(values) < 2:
            continue
        anchor = values[0]
        with np.errstate(over="ignore", invalid="ignore"):
            delta = values - anchor
        if not np.isfinite(delta).all():
            pre_scales[index] = np.max(np.abs(values))
            values = values / pre_scales[index]
            anchor = values[0]
            delta = values - anchor
        scale = np.max(np.abs(delta))
        if scale == 0:
            continue
        anchors[index] = anchor
        scales[index] = scale
        means[index] = (delta / scale).mean(dtype=np.float64)
        active[index] = True
        # pandas' unscaled Welford accumulators can overflow/underflow at
        # these magnitudes. Preserve its actual result through targeted
        # recomputation rather than silently changing NaN behavior.
        max_abs = np.max(np.abs(values)) * pre_scales[index]
        extreme[index] = (max_abs > 1e75) or (scale * pre_scales[index] < 1e-75)
    return anchors, scales, pre_scales, means, active, extreme


def _accumulate(frame, positions, parameters, xp, row_batch, column_batch):
    """Stream bounded host chunks; accumulate four float64 m-by-m matrices."""
    anchors, scales, pre_scales, means, _, _ = parameters
    anchors = anchors[positions]
    scales = scales[positions]
    pre_scales = pre_scales[positions]
    means = means[positions]
    width = len(positions)
    n = xp.zeros((width, width), dtype=xp.float64)
    p = xp.zeros_like(n)
    s = xp.zeros_like(n)
    q = xp.zeros_like(n)
    peak_pool_bytes = 0
    gpu = xp is not np
    for start in range(0, len(frame), row_batch):
        # Only this row batch is copied, avoiding a whole-frame float64
        # materialization from a heterogeneous pandas block manager.
        host = frame.iloc[start : start + row_batch, positions].to_numpy(
            dtype=np.float64, copy=True, na_value=np.nan
        )
        finite = np.isfinite(host)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            host /= pre_scales
            host -= anchors
            host /= scales
            host -= means
        host[~finite] = np.nan
        x = xp.asarray(host)
        del host, finite
        mask = xp.isfinite(x).astype(xp.float64)
        x = xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        squared = x * x
        for left in range(0, width, column_batch):
            li = slice(left, min(left + column_batch, width))
            for right in range(0, width, column_batch):
                ri = slice(right, min(right + column_batch, width))
                n[li, ri] += mask[:, li].T @ mask[:, ri]
                p[li, ri] += x[:, li].T @ x[:, ri]
                s[li, ri] += x[:, li].T @ mask[:, ri]
                q[li, ri] += squared[:, li].T @ mask[:, ri]
        if gpu:
            peak_pool_bytes = max(peak_pool_bytes, xp.get_default_memory_pool().total_bytes())
        del x, mask, squared
    if gpu:
        xp.cuda.get_current_stream().synchronize()
        return tuple(xp.asnumpy(value) for value in (n, p, s, q)), peak_pool_bytes
    return (n, p, s, q), peak_pool_bytes


def _finish(frame, positions, parameters, stats, min_periods, threshold):
    n, p, s, q = stats
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        correction = s * s / n
        variance = q - correction
        covariance = p - s * s.T / n
        denominator = np.sqrt(np.maximum(variance, 0)) * np.sqrt(
            np.maximum(variance.T, 0)
        )
        corr = covariance / denominator
    valid = (n >= max(2, min_periods)) & (variance > 0) & (variance.T > 0)
    corr[~valid] = np.nan
    np.clip(corr, -1, 1, out=corr)
    # Global translation cannot prevent cancellation when a column is
    # nearly constant only on its pairwise-overlap subset.
    tolerance = 128 * np.finfo(np.float64).eps * (np.abs(q) + np.abs(correction))
    suspect = (n >= max(2, min_periods)) & (
        (variance <= tolerance) | (variance.T <= tolerance.T)
    )
    if threshold is not None:
        suspect |= np.isfinite(corr) & (np.abs(np.abs(corr) - threshold) <= 1e-10)
    extreme = parameters[5][positions]
    suspect |= (n >= max(2, min_periods)) & (extreme[:, None] | extreme[None, :])
    suspect |= (n >= max(2, min_periods)) & ~np.isfinite(corr)
    corrected = 0
    for left, right in zip(*np.where(np.triu(suspect))):
        indexes = [positions[left], positions[right]]
        # A two-column pandas calculation uses the reference Welford
        # semantics and needs only a bounded O(n) temporary per risky pair.
        value = frame.iloc[:, indexes].corr(min_periods=min_periods).iloc[0, 1]
        corr[left, right] = corr[right, left] = value
        corrected += 1
    # Normalize tiny GEMM asymmetry without altering missingness; NaNs
    # represent constant columns, too few observations or missing overlaps.
    corr = (corr + corr.T) * 0.5
    result = np.full((frame.shape[1], frame.shape[1]), np.nan, dtype=np.float64)
    result[np.ix_(positions, positions)] = corr
    return pd.DataFrame(result, index=frame.columns, columns=frame.columns), corrected


def pairwise_pearson(
    frame,
    *,
    min_periods=1,
    row_batch=8192,
    column_batch=None,
    memory_budget_mb=256,
    threshold=0.95,
    backend="gpu",
    return_report=False,
):
    """Equivalent Pearson correlations, with optional GPU acceleration.

    ``backend='numpy'`` exists only for prototype validation. GPU mode
    raises a useful ImportError if CuPy is absent and never silently falls
    back. A private memory pool is bounded and fully released on return.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a numeric pandas DataFrame.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError("All columns must be numeric.")
    if row_batch < 1 or memory_budget_mb < 16:
        raise ValueError("row_batch must be positive and memory_budget_mb >= 16.")
    if min_periods < 0:
        raise ValueError("min_periods must be nonnegative.")
    if backend == "gpu":
        try:
            import cupy as xp
        except ImportError as exc:
            raise ImportError("GPU correlation requires a compatible CuPy installation.") from exc
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA GPU is available.")
        pool = xp.cuda.MemoryPool()
        budget = int(memory_budget_mb * 1024**2)
        pool.set_limit(size=budget)
        allocation_context = xp.cuda.using_allocator(pool.malloc)
    elif backend == "numpy":
        xp = np
        pool = None
        budget = int(memory_budget_mb * 1024**2)
        allocation_context = nullcontext()
    else:
        raise ValueError("backend must be 'gpu' or 'numpy'.")
    started = time.perf_counter()
    report = {
        "backend": backend,
        "shape": list(frame.shape),
        "memory_budget_mb": memory_budget_mb,
        "dtype": "float64",
    }
    try:
        parameters = _column_parameters(frame)
        positions = np.flatnonzero(parameters[4])
        width = len(positions)
        if width:
            # Four accumulators and conservative temporary/workspace
            # allowance; the row chunk is never retained across batches.
            reserve = 10 * 8 * width * width
            available = budget - reserve
            if available < 40 * width:
                raise ValueError("Memory budget is too small for the correlation matrix.")
            effective_rows = min(row_batch, max(1, available // (40 * width)))
            effective_columns = width if column_batch is None else min(column_batch, width)
            if effective_columns < 1:
                raise ValueError("column_batch must be positive.")
            with allocation_context:
                stats, _ = _accumulate(
                    frame, positions, parameters, xp, effective_rows, effective_columns
                )
            report["peak_private_pool_mb"] = pool.total_bytes() / 1024**2 if pool else 0.0
            result, corrected = _finish(
                frame, positions, parameters, stats, min_periods, threshold
            )
            report.update(
                active_columns=width,
                row_batch=effective_rows,
                column_batch=effective_columns,
                corrected_pairs=corrected,
            )
        else:
            result = pd.DataFrame(np.nan, index=frame.columns, columns=frame.columns)
            report.update(active_columns=0, corrected_pairs=0, peak_private_pool_mb=0.0)
    finally:
        if pool is not None:
            xp.cuda.get_current_stream().synchronize()
            pool.free_all_blocks()
            report["private_pool_retained_bytes"] = pool.total_bytes()
    report["seconds"] = time.perf_counter() - started
    return (result, report) if return_report else result
