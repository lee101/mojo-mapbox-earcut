# mojo-mapbox-earcut

`mojo-mapbox-earcut` is a Mojo implementation of Earcut-style polygon triangulation for Python. It triangulates simple 2D polygons, including concave outlines and multiple holes, and exposes the same four typed entry points as [`mapbox_earcut`](https://pypi.org/project/mapbox-earcut/).

## Install

```bash
pixi install
pixi run build
```

`pixi install` supplies Mojo, NumPy, pytest, and `mapbox_earcut`, which is used by the parity tests and benchmark. This repository currently ships as a source checkout rather than a PyPI package; run examples through `pixi run python` from the checkout.

## Usage

`ring_end_indices` contains the exclusive end vertex index of the outer ring followed by every hole.

```python
import numpy as np
import mojo_mapbox_earcut as earcut

vertices = np.array([
    [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],  # outer ring
    [2.0, 2.0], [2.0, 8.0], [8.0, 8.0], [8.0, 2.0],      # hole
])
triangles = earcut.triangulate_float64(vertices, np.array([4, 8], dtype=np.uint32))
print(triangles.reshape(-1, 3))
```

The returned `uint32` triples index `vertices`. Available functions are `triangulate_float32`, `triangulate_float64`, `triangulate_int32`, and `triangulate_int64`.

## Coverage

Covered and tested: simple polygons in either winding direction, concave outlines, multiple holes, and collinear or repeated vertices. The public Python surface has the four typed functions provided by upstream `mapbox_earcut`: `triangulate_float32`, `triangulate_float64`, `triangulate_int32`, and `triangulate_int64`. Tests exercise every entry point on convex polygons and polygons with a hole; the float64 tests additionally exercise the other listed shapes and SIMD tail lengths.

Not covered: self-intersecting polygons, arbitrary 3D projection, and upstream's z-order spatial index. Input must be a planar `(n, 2)` numeric array and strictly increasing integer ring ends that fit `uint32`; coordinate conversions that would lose precision are rejected. Self-intersections do not have a well-defined filled region without a separate fill-rule operation. Omitting the z-order index trades large-polygon throughput for a compact single Mojo compilation unit.

## How it works

The kernel stores polygon vertices in one contiguous `float64` `(n, 2)` array. Python validates dimensions, integer ring ends, and lossless conversion before passing raw buffer addresses to one C-ABI Mojo export through `ctypes`; the NumPy arrays remain live for the full native call. Caller-owned `Int64` arrays hold `prev`, `next`, and source-index fields for a circular doubly linked list; contiguous initialization uses SIMD with a scalar tail. Joining a hole duplicates its two bridge endpoints in that scratch space; clipped triangle indices are written into a caller-owned output buffer and returned as `uint32`.

No kernel allocation crosses the FFI boundary, and the shared library is built as `dist/libmojo-mapbox-earcut.so` by `pixi run build`.

Triangulation is branch-heavy and pointer-chasing, with insufficient arithmetic intensity for a GPU path; CPU is the only execution path.

## Verification

```bash
pixi run test
pixi run bench
```

The pytest suite compares triangle count, index bounds, and covered area against the real `mapbox_earcut` package across typed entry points, concavity, holes, degeneracies, and SIMD-tail sizes. It also checks invalid layouts, non-contiguous NumPy inputs, precision-loss rejection, and null C-ABI arguments.

## Benchmarks

Measured on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39 (x86_64)` using `pixi run bench` on 2026-08-02. Lower is better; speedup is `mapbox-earcut / Mojo`.

| kernel | Mojo | mapbox-earcut | speedup |
| --- | ---: | ---: | ---: |
| concave star (800 vertices) | 1.028 ms | 0.203 ms | 0.20x |
| rectangle with 6 holes (28 vertices) | 0.079 ms | 0.007 ms | 0.09x |

This implementation is slower than upstream's optimized native implementation on both measured kernels. Its value here is a standalone, inspectable Mojo kernel with a compatible Python API; the missing z-order index is the principal reason the gap widens on large concave rings.

MIT.
