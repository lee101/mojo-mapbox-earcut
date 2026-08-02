"""Mojo implementation of the typed ``mapbox_earcut`` triangulation API."""

from __future__ import annotations

import numpy as np

from ._lib import lib

__all__ = ["triangulate_float32", "triangulate_float64", "triangulate_int32", "triangulate_int64"]


def _triangulate(vertices, ring_end_indices, dtype: np.dtype) -> np.ndarray:
    raw_points = np.asarray(vertices)
    if raw_points.ndim != 2 or raw_points.shape[1] != 2:
        raise ValueError("vertices must be a two-dimensional array with shape (n, 2)")
    if not (np.issubdtype(raw_points.dtype, np.integer) or np.issubdtype(raw_points.dtype, np.floating)):
        raise TypeError("vertices must have an integer or floating-point dtype")
    try:
        points = np.asarray(raw_points, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"vertices cannot be represented exactly as {dtype}") from exc
    # The public typed API must not quietly narrow coordinates. The kernel itself
    # is float64, so also verify that integer coordinates survive that widening.
    if not np.array_equal(points.astype(raw_points.dtype), raw_points, equal_nan=True):
        raise ValueError(f"vertices cannot be represented exactly as {dtype}")
    xy = np.asarray(points, dtype=np.float64)
    if np.issubdtype(points.dtype, np.integer) and not np.array_equal(xy.astype(points.dtype), points):
        raise ValueError("integer vertices cannot be represented exactly by the float64 kernel")
    points = np.ascontiguousarray(points)
    xy = np.ascontiguousarray(xy)

    raw_ends = np.asarray(ring_end_indices)
    if raw_ends.ndim != 1:
        raise ValueError("ring_end_indices must be one-dimensional")
    if not np.issubdtype(raw_ends.dtype, np.integer):
        raise TypeError("ring_end_indices must have an integer dtype")
    if np.any(raw_ends < 0) or np.any(raw_ends > np.iinfo(np.uint32).max):
        raise ValueError("ring_end_indices must fit in uint32")
    ends = np.ascontiguousarray(raw_ends, dtype=np.uint32)
    n = len(points)
    if not len(ends):
        return np.empty(0, dtype=np.uint32)
    if np.any(ends <= 0) or np.any(ends[1:] <= ends[:-1]) or ends[-1] != n:
        raise ValueError("ring_end_indices must be strictly increasing and end at len(vertices)")
    # Each hole bridge duplicates two vertices.  A polygon with h holes has
    # n + 2h - 2 triangles after bridges are introduced.
    output = np.empty(max(0, 3 * (n + 2 * (len(ends) - 1) - 2)), dtype=np.int64)
    capacity = n + 2 * len(ends)
    scratch = np.empty((3, capacity), dtype=np.int64)
    holes = np.empty(max(1, len(ends) - 1), dtype=np.int64)
    written = lib().mme_triangulate_f64(
        xy.ctypes.data, ends.ctypes.data, len(ends), n, output.ctypes.data, len(output),
        scratch[0].ctypes.data, scratch[1].ctypes.data, scratch[2].ctypes.data, capacity,
        holes.ctypes.data, len(holes),
    )
    if written == -1:
        raise ValueError("invalid polygon layout")
    if written < 0:
        raise ValueError("triangulation did not complete; input may be self-intersecting")
    return output[:written].astype(np.uint32, copy=False)


def triangulate_float32(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.float32)


def triangulate_float64(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.float64)


def triangulate_int32(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.int32)


def triangulate_int64(vertices, ring_end_indices):
    return _triangulate(vertices, ring_end_indices, np.int64)
