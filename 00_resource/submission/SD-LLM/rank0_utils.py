# rank0_utils.py
# Minimal helper to ensure only rank 0 prints/logs.
# Usage in your script (very top, before any prints/logging):
#   from rank0_utils import patch_print, configure_logging_for_rank0
#   patch_print()                      # make built-in print no-op on nonzero ranks
#   configure_logging_for_rank0()      # quiet libraries on nonzero ranks

from __future__ import annotations

import os
import sys
import builtins
import contextlib
from typing import Any, Callable, Optional

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore

__all__ = [
    "get_world_size",
    "get_global_rank",
    "is_rank0",
    "print0",
    "patch_print",
    "configure_logging_for_rank0",
    "suppress_stdout_stderr_if_not_rank0",
]

# -------- Rank helpers --------

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default

def _dist_is_initialized() -> bool:
    return (dist is not None) and getattr(dist, "is_available", lambda: False)() and dist.is_initialized()

def get_world_size() -> int:
    if _dist_is_initialized():
        try:
            return dist.get_world_size()
        except Exception:
            pass
    # Fallbacks for launchers
    ws = _env_int("WORLD_SIZE", 1)
    if ws > 1:
        return ws
    ws = _env_int("SLURM_NTASKS", 1)
    return ws if ws > 0 else 1

def get_global_rank() -> int:
    if _dist_is_initialized():
        try:
            return dist.get_rank()
        except Exception:
            pass
    # Common env fallbacks: torchrun/DeepSpeed, SLURM
    for k in ("RANK", "SLURM_PROCID"):
        r = os.environ.get(k)
        if r is not None:
            try:
                return int(r)
            except ValueError:
                continue
    return 0

def is_rank0() -> bool:
    return get_global_rank() == 0

# -------- Printing --------

_ORIG_PRINT: Optional[Callable[..., None]] = None

def print0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0 (always available even if patch_print is not used)."""
    if is_rank0():
        builtins.print(*args, **kwargs)

def patch_print(force: bool = False) -> None:
    """
    Monkey-patch built-in print to be a no-op on nonzero ranks.
    Call this once, near the top of your main script.
    If `force=True`, re-patch even if already patched.
    """
    global _ORIG_PRINT
    if _ORIG_PRINT is None or force:
        _ORIG_PRINT = builtins.print

        def _ranked_print(*args: Any, **kwargs: Any) -> None:
            if is_rank0():
                _ORIG_PRINT(*args, **kwargs)  # type: ignore[misc]

        builtins.print = _ranked_print  # type: ignore[assignment]

# -------- Logging / library verbosity --------

def configure_logging_for_rank0(
    transformers_level: str = "error",
    datasets_level: str = "error",
    tokenizers_parallelism: Optional[bool] = False,
) -> None:
    """
    Quiet common libraries on nonzero ranks.
    - Sets transformers & datasets loggers to ERROR on nonzero ranks.
    - Optionally disables tokenizers parallelism (avoids extra noise).
    Safe to call on all ranks; rank 0 keeps defaults.
    """
    nonzero = not is_rank0()

    # Avoid parallel tokenizers chatter
    if tokenizers_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"

    # Python logging root: keep as-is on rank0; raise threshold on others.
    import logging

    if nonzero:
        logging.getLogger().setLevel(logging.WARNING)

    # Hugging Face transformers
    try:
        from transformers.utils import logging as hf_logging
        if nonzero:
            level = transformers_level.lower()
            lvl = {
                "critical": hf_logging.CRITICAL,
                "error": hf_logging.ERROR,
                "warning": hf_logging.WARNING,
                "info": hf_logging.INFO,
                "debug": hf_logging.DEBUG,
            }.get(level, hf_logging.ERROR)
            hf_logging.set_verbosity(lvl)
            hf_logging.disable_default_handler()
            hf_logging.enable_propagation()
    except Exception:
        pass

    # 🤗 Datasets
    try:
        import datasets
        if nonzero:
            datasets.logging.set_verbosity_error()
    except Exception:
        pass

# -------- Stdout/stderr suppression (optional) --------

@contextlib.contextmanager
def suppress_stdout_stderr_if_not_rank0():
    """
    Context manager to fully silence stdout/stderr on nonzero ranks (e.g., around noisy sections).
    """
    if is_rank0():
        yield
        return
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        try:
            devnull.close()
        except Exception:
            pass
