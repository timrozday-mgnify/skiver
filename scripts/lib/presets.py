"""Named platform presets for the generative error model.

A preset maps a friendly platform name (e.g. ``hq-illumina``, ``ont``) to a
bundled trained ``.pt`` artifact under ``context_error_models/``.  The registry
lives in ``context_error_models/presets.json`` so the mapping can be edited
without touching code.  These presets replace genome-blender's former built-in
HMM platform profiles.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# scripts/lib/presets.py -> repo root is two levels up from scripts/.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "context_error_models"
_REGISTRY_PATH = _MODELS_DIR / "presets.json"


@lru_cache(maxsize=1)
def _registry() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(presets, aliases)`` from the registry file."""
    with open(_REGISTRY_PATH) as fh:
        data = json.load(fh)
    return data.get("presets", {}), data.get("aliases", {})


def available_presets() -> list[str]:
    """Return the sorted list of preset and alias names callers may pass."""
    presets, aliases = _registry()
    return sorted({*presets, *aliases})


def resolve_preset(name: str) -> Path:
    """Resolve a preset / alias name to a bundled artifact path.

    Args:
        name: Preset name (e.g. ``ont``) or alias (e.g. ``nanopore``).

    Returns:
        Absolute path to the bundled ``.pt`` artifact.

    Raises:
        KeyError: If the name is not a known preset or alias.
        FileNotFoundError: If the registered artifact is missing on disk.
    """
    presets, aliases = _registry()
    resolved = aliases.get(name, name)
    if resolved not in presets:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {', '.join(available_presets())}"
        )
    path = _MODELS_DIR / presets[resolved]
    if not path.exists():
        raise FileNotFoundError(f"Preset {name!r} -> missing artifact {path}")
    return path
