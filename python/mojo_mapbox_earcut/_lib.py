"""ctypes loader for the Mojo Earcut kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MAPBOX_EARCUT_LIB") or os.path.join(ROOT, "dist", "libmojo-mapbox-earcut.so")
I = ctypes.c_int64


def build() -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo is not on PATH; use pixi run build")
    proc = subprocess.run([mojo, "build", "--emit", "shared-lib", sources[0], "-o", LIB], capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        _lib.mme_triangulate_f64.argtypes = [I] * 12
        _lib.mme_triangulate_f64.restype = I
    return _lib
