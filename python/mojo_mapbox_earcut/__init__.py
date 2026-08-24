"""Mojo implementation of the typed ``mapbox_earcut`` triangulation API."""

from __future__ import annotations

import numpy as np

from ._lib import lib

__all__ = ["triangulate_float32", "triangulate_float64", "triangulate_int32", "triangulate_int64"]

_FLOAT64 = np.dtype(np.float64)
_UINT32 = np.dtype(np.uint32)
_UINT32_MAX = np.iinfo(np.uint32).max


def _triangulate(vertices, ring_end_indices, dtype: np.dtype) -> np.ndarray:
    raw_points = np.asarray(vertices)
    if raw_points.ndim != 2 or raw_points.shape[1] != 2:
        raise ValueError("vertices must be a two-dimensional array with shape (n, 2)")
    if raw_points.dtype.kind not in "iuf":
        raise TypeError("vertices must have an integer or floating-point dtype")
    target_dtype = np.dtype(dtype)
    if raw_points.dtype == target_dtype:
        points = raw_points
    else:
        try:
            points = np.asarray(raw_points, dtype=target_dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"vertices cannot be represented exactly as {dtype}") from exc
        if not np.array_equal(points.astype(raw_points.dtype), raw_points, equal_nan=True):
            raise ValueError(f"vertices cannot be represented exactly as {dtype}")
    if target_dtype == _FLOAT64:
        xy = np.ascontiguousarray(points)
    else:
        xy = np.asarray(points, dtype=np.float64)
        if points.dtype.kind in "iu" and not np.array_equal(xy.astype(points.dtype), points):
            raise ValueError("integer vertices cannot be represented exactly by the float64 kernel")
        xy = np.ascontiguousarray(xy)

    raw_ends = np.asarray(ring_end_indices)
    if raw_ends.ndim != 1:
        raise ValueError("ring_end_indices must be one-dimensional")
    if raw_ends.dtype.kind not in "iu":
        raise TypeError("ring_end_indices must have an integer dtype")
    if raw_ends.dtype == _UINT32:
        ends = np.ascontiguousarray(raw_ends)
    else:
        if raw_ends.size and (int(raw_ends.min()) < 0 or int(raw_ends.max()) > _UINT32_MAX):
            raise ValueError("ring_end_indices must fit in uint32")
        ends = np.ascontiguousarray(raw_ends, dtype=np.uint32)
    n = len(points)
    ring_count = len(ends)
    if not ring_count:
        return np.empty(0, dtype=np.uint32)
    previous_end = 0
    for ring_end in ends:
        current_end = int(ring_end)
        if current_end <= previous_end:
            raise ValueError("ring_end_indices must be strictly increasing and end at len(vertices)")
        previous_end = current_end
    if previous_end != n:
        raise ValueError("ring_end_indices must be strictly increasing and end at len(vertices)")
    # Each hole bridge duplicates two vertices.  A polygon with h holes has
    # n + 2h - 2 triangles after bridges are introduced.
    output = np.empty(max(0, 3 * (n + 2 * (ring_count - 1) - 2)), dtype=np.uint32)
    capacity = n + 2 * ring_count
    scratch_rows = 6 if n > 80 else 3
    scratch = np.empty((scratch_rows, capacity), dtype=np.int64)
    holes = np.empty(max(1, ring_count - 1), dtype=np.int64)
    scratch_addr = scratch.ctypes.data
    written = lib().mme_triangulate_f64(
        xy.ctypes.data, ends.ctypes.data, ring_count, n, output.ctypes.data, len(output),
        scratch_addr, capacity, holes.ctypes.data, len(holes),
    )
    if written == -1:
        raise ValueError("invalid polygon layout")
    if written < 0:
        raise ValueError("triangulation did not complete; input may be self-intersecting")
    return output[:written]


def triangulate_float32(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.float32)


def triangulate_float64(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.float64)


def triangulate_int32(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.int32)


def triangulate_int64(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.int64)
