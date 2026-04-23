"""
Preload NVIDIA NPP shared libraries so torchcodec can find them.

torch's bundled CUDA preloader (see torch/__init__.py::_preload_cuda_deps)
only covers the libs torch itself uses — cuBLAS, cuDNN, cuFFT, cuRAND,
nvjitlink, cusparse, cusparselt, cusolver, nccl, nvshmem, cufile, nvtx.
NPP is absent. torchcodec, however, links against libnppicc at load time.
Inside a bare CUDA container (no /usr/local/cuda on the loader path) this
means torchaudio.load -> torchcodec fails with

    OSError: libnppicc.so.13: cannot open shared object file

even though `nvidia-npp` is installed, because its lib dir in
site-packages is not on the dynamic loader's search path.

This module walks sys.path, finds the installed NPP .so's, and loads them
RTLD_GLOBAL so torchcodec's later dlopen() sees them as already-resolved.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys


def apply() -> None:
    # libnppc is the NPP core; other libnpp*.so depend on it, so load it first.
    for name in ("libnppc.so.*",):
        _load_matching(name)
    _load_matching("libnpp*.so.*")


def _load_matching(pattern: str) -> None:
    for path in sys.path:
        lib_dir = os.path.join(path, "nvidia", "cu13", "lib")
        if not os.path.isdir(lib_dir):
            continue
        for so in sorted(glob.glob(os.path.join(lib_dir, pattern))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Best effort. If the symbol is truly missing, torchcodec will
                # raise its own, clearer error when the app later uses it.
                pass
