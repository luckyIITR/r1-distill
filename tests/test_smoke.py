"""Smoke tests for environment setup."""
from __future__ import annotations


def test_python_version() -> None:
    import sys
    assert sys.version_info >= (3, 10)


def test_torch_available() -> None:
    import torch
    assert torch.__version__ >= "2.4"


def test_transformers_available() -> None:
    import transformers
    assert transformers.__version__ >= "4.46"


def test_swift_importable() -> None:
    import swift  # noqa: F401


def test_cuda_visible_when_expected() -> None:
    import os
    import torch
    if os.environ.get("CHECK_CUDA") == "1":
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1