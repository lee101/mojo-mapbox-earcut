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

Covered and tested: simple polygons in either winding direction, concave outlines, multiple holes, and collinear or repeated vertices. The public Python surface has the four typed functions provided by upstream `mapbox_earcut`: `triangulate_float32`, `triangulate_float64`, `triangulate_int32`, and `triangulate_int64`. Tests exercise every entry point on convex polygons and polygons with a hole; the float64 tests additionally exercise the other listed shapes, SIMD tail lengths, both sides of the spatial-index threshold, and indexed polygons with holes.

Not covered: self-intersecting polygons, arbitrary 3D projection, and upstream's final degeneracy-recovery and polygon-splitting passes. Input must be a planar `(n, 2)` numeric array and strictly increasing integer ring ends that fit `uint32`; coordinate conversions that would lose precision are rejected. Self-intersections do not have a well-defined filled region without a separate fill-rule operation.

## How it works

The kernel stores polygon vertices in one contiguous `float64` `(n, 2)` array. Python validates dimensions, integer ring ends, and lossless conversion before passing raw buffer addresses to one C-ABI Mojo export through `ctypes`; already-contiguous `float64` vertices and `uint32` ring ends cross that boundary without copies. Caller-owned `Int64` arrays hold linked-list and spatial-index fields. Contiguous ring initialization and signed-area accumulation use native-width SIMD with scalar tails. Polygons above 80 vertices build a z-order linked index so ear tests inspect only spatially relevant candidates. Joining a hole duplicates its two bridge endpoints in scratch space; clipped triangle indices are written directly to a caller-owned `uint32` output buffer without a result conversion.

No kernel allocation crosses the FFI boundary, and the shared library is built as `dist/libmojo-mapbox-earcut.so` by `pixi run build`.

Ear clipping is branch-heavy, pointer-chasing, and sequential: removing one ear changes the candidates for the next iteration. It has insufficient independent work for CPU thread launch overhead and far less than the roughly two flops per byte needed to justify host/device transfers, so there is no parallel or GPU path. CPU is the only execution path.

## Verification

```bash
pixi run test
pixi run bench
```

The pytest suite compares triangle count, index bounds, and covered area against the real `mapbox_earcut` package across typed entry points, concavity, holes, degeneracies, and SIMD-tail sizes. It also checks invalid layouts, non-contiguous NumPy inputs, precision-loss rejection, and null C-ABI arguments.

## Benchmarks

Measured on `Linux-6.8.0-136-generic-x86_64-with-glibc2.39 (x86_64)` using `pixi run bench` on 2026-08-24. Lower is better; speedup is `mapbox-earcut / Mojo`.

| kernel | Mojo | mapbox-earcut | speedup |
| --- | ---: | ---: | ---: |
| concave star (800 vertices) | 0.231 ms | 0.182 ms | 0.79x |
| rectangle with 6 holes (28 vertices) | 0.024 ms | 0.006 ms | 0.27x |

This implementation remains slower than upstream's optimized native implementation on both measured kernels. The z-order index closes most of the large-ring gap; fixed Python validation, allocation, and FFI costs remain most visible on the 28-vertex case.

MIT.
