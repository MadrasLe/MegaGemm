# -*- coding: utf-8 -*-
"""
MegaGemm -- Environment-aware build script.

Tries to compile CUDA extensions (RMSNorm + RoPE) when nvcc and a compatible
torch+CUDA are available.  When they aren't (AMD GPU, CPU-only machine,
Colab without nvcc, etc.) the build silently skips the native extensions
and the package falls back to Triton / pure-PyTorch kernels at runtime.

This means ``pip install -e .`` always succeeds, on any machine.  CUDA
kernels are a *bonus* that the package auto-detects at import time via
``import rmsnorm_cuda_ops``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Optional

from setuptools import Extension, setup

# ---------------------------------------------------------------------------
# CUDA environment detection
# ---------------------------------------------------------------------------

def _normalize_cuda(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    parts = v.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else v


def _detect_nvcc_cuda() -> Optional[str]:
    """Return the CUDA version reported by nvcc, or None."""
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, text=True,
        )
    except Exception:
        return None
    m = re.search(r"release\s+(\d+\.\d+)", out)
    return _normalize_cuda(m.group(1) if m else None)


def _detect_torch_cuda() -> Optional[str]:
    """Return the CUDA version that torch was compiled with, or None."""
    try:
        import torch
    except ImportError:
        return None
    return _normalize_cuda(torch.version.cuda)


def _can_build_cuda() -> bool:
    """
    Decide whether we should attempt to compile CUDA extensions.

    Returns True only when:
      1. nvcc is reachable on $PATH
      2. torch is installed with CUDA support
      3. The CUDA major versions match (e.g. both 12.x)

    Any mismatch prints a friendly diagnostic instead of crashing.
    """
    nvcc_cuda = _detect_nvcc_cuda()
    torch_cuda = _detect_torch_cuda()

    if nvcc_cuda is None:
        print(
            "[MegaGemm] nvcc not found -- skipping CUDA extension build.\n"
            "   The package will use Triton / PyTorch fallback kernels.\n"
            "   To enable CUDA kernels, install the CUDA Toolkit and rebuild."
        )
        return False

    if torch_cuda is None:
        print(
            "[MegaGemm] torch is not installed or has no CUDA support --\n"
            "   skipping CUDA extension build."
        )
        return False

    # Compare major version (e.g. "12" from "12.4")
    nvcc_major = nvcc_cuda.split(".")[0]
    torch_major = torch_cuda.split(".")[0]

    if nvcc_major != torch_major:
        print(
            f"[MegaGemm] CUDA version mismatch -- nvcc={nvcc_cuda}, "
            f"torch={torch_cuda}.\n"
            f"   Skipping CUDA extension build to avoid ABI issues.\n"
            f"   To fix: install a torch wheel matching your CUDA toolkit,\n"
            f"   or run: python scripts/install_smart.py --editable"
        )
        return False

    if nvcc_cuda != torch_cuda:
        # Minor version mismatch -- usually fine, just warn
        print(
            f"[MegaGemm] minor CUDA mismatch (nvcc={nvcc_cuda}, "
            f"torch={torch_cuda}), building anyway."
        )

    return True


# ---------------------------------------------------------------------------
# Extension modules
# ---------------------------------------------------------------------------

ext_modules = []
cmdclass = {}

# Allow explicit skip via env var (useful for CI / sdist)
_force_skip = os.environ.get("MEGAGEMM_SKIP_CUDA", "0") == "1"
_force_skip_native = os.environ.get("MEGAGEMM_SKIP_NATIVE", "0") == "1"
_build_only_rmsnorm_cuda = (
    os.environ.get("MEGAGEMM_BUILD_ONLY_RMSNORM_CUDA", "0") == "1"
)
_build_only_cublaslt = (
    os.environ.get("MEGAGEMM_BUILD_ONLY_CUBLASLT", "0") == "1"
)

# Pure CPython helper for TTP packet receive.  This intentionally does not use
# torch.utils.cpp_extension so importing it never depends on libtorch/libc10.
if (
    not _force_skip_native
    and not _build_only_rmsnorm_cuda
    and not _build_only_cublaslt
):
    ext_modules.append(
        Extension(
            "megagemm_ttp_native",
            ["src/ttp_native.c"],
            libraries=["Ws2_32"] if sys.platform.startswith("win") else [],
        )
    )
    print("[MegaGemm] Building native TTP receive helper.")
elif _force_skip_native:
    print("[MegaGemm] MEGAGEMM_SKIP_NATIVE=1 -- skipping all native extensions.")
else:
    print("[MegaGemm] Skipping unrelated native extensions for focused CUDA build.")

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    # C++ decode orchestration: full-attention helpers plus native CUDA-graph
    # burst replay.  This remains a CppExtension because graph execution is
    # provided by the torch CUDAGraph binding; no project CUDA source is needed.
    if (
        not _force_skip_native
        and not _build_only_rmsnorm_cuda
        and not _build_only_cublaslt
    ):
        ext_modules.append(
            CppExtension(
                "megagemm_decode_ops",
                ["src/decode_loop.cpp"],
            )
        )
        cmdclass = {"build_ext": BuildExtension}
        print("[MegaGemm] Building CPU extension (decode loop helper).")

    if not _force_skip_native and not _force_skip and _can_build_cuda():
        cmdclass = {"build_ext": BuildExtension}
        if _build_only_cublaslt:
            ext_modules.append(
                CUDAExtension(
                    "megagemm_cublaslt_ops",
                    [
                        "pytorch_binding/cublaslt_binding.cpp",
                        "src/mlp_prefill_kernel.cu",
                    ],
                    libraries=["cublas", "cublasLt"],
                ),
            )
            print("[MegaGemm] Building only the focused cuBLASLt extension.")
        else:
            ext_modules.append(
                CUDAExtension(
                    "rmsnorm_cuda_ops",
                    [
                        "pytorch_binding/binding.cpp",
                        "src/rmsnorm_kernel.cu",
                        "src/rope_kernel.cu",
                        "src/mlp_prefill_kernel.cu",
                    ],
                    libraries=["cublas", "cublasLt"],
                ),
            )
        if not _build_only_rmsnorm_cuda and not _build_only_cublaslt:
            ext_modules.append(
                CUDAExtension(
                    "sparse24_cuda_ops",
                    [
                        "pytorch_binding/sparse24_binding.cpp",
                        "src/sparse24_fp16_kernel.cu",
                    ],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": ["-O3", "-lineinfo"],
                    },
                ),
            )
            print("[MegaGemm] Building CUDA extensions (RMSNorm + RoPE + standalone FP16 2:4 mma.sp).")
        elif _build_only_rmsnorm_cuda:
            print("[MegaGemm] Building only rmsnorm_cuda_ops as requested.")
    elif _force_skip_native:
        pass
    elif _force_skip:
        print("[MegaGemm] MEGAGEMM_SKIP_CUDA=1 -- skipping CUDA build.")
except ImportError:
    print(
        "[MegaGemm] torch.utils.cpp_extension not available --\n"
        "   skipping torch-backed native extension build."
    )


# ---------------------------------------------------------------------------
# Setup -- metadata lives in pyproject.toml (PEP 621)
# ---------------------------------------------------------------------------

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
