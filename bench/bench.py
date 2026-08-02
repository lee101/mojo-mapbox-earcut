"""Measure the Mojo kernel and mapbox_earcut on identical polygon buffers."""

from __future__ import annotations

import platform
import time

import mapbox_earcut as upstream
import numpy as np

import mojo_mapbox_earcut as mojo


def best_time(fn, repetitions=5):
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def star(n):
    theta = np.arange(n) * (2 * np.pi / n)
    radius = np.where(np.arange(n) % 2, 40.0, 100.0)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def with_holes():
    outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    holes = []
    for x, y in ((10, 10), (35, 10), (60, 10), (10, 45), (35, 45), (60, 45)):
        holes.extend([[x, y], [x, y + 15], [x + 15, y + 15], [x + 15, y]])
    vertices = np.vstack((outer, np.asarray(holes, dtype=np.float64)))
    return vertices, np.arange(4, len(vertices) + 1, 4, dtype=np.uint32)


def main():
    cases = [("concave star (800 vertices)", star(800), np.array([800], dtype=np.uint32))]
    holes, ends = with_holes()
    cases.append(("rectangle with 6 holes (28 vertices)", holes, ends))
    print(f"Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print()
    print("| kernel | Mojo | mapbox-earcut | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, vertices, ring_ends in cases:
        mojo_seconds = best_time(lambda: mojo.triangulate_float64(vertices, ring_ends))
        upstream_seconds = best_time(lambda: upstream.triangulate_float64(vertices, ring_ends))
        print(f"| {name} | {mojo_seconds * 1e3:.3f} ms | {upstream_seconds * 1e3:.3f} ms | {upstream_seconds / mojo_seconds:.2f}x |")


if __name__ == "__main__":
    main()
