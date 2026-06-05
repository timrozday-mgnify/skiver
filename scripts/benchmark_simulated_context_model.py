#!/usr/bin/env python3
"""Generate synthetic reads with genome-blender and test parameter recovery."""
from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import concurrent.futures
import functools
import json
import logging
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from lib.context_error_models import (
    ContextCounts,
    ContextLengthScreenCounts,
    aggregate_context_length_screen_counts,
    calibrate_to_rate,
    compute_marginal_error_rate,
    compute_marginal_weibull,
    fit_bayesian_and_test,
    fit_and_test,
    subsample_context_counts,
)
from lib.encoding import BASE_TO_IDX, NUM_ERROR_TYPES, encode_error_type
from fit_quality_calibration_model import (
    CalibrationCounts,
    _counts_table,
    _fit_model,
    _fit_summaries_to_json,
    fit_quality_calibration,
)
from simulate_errors import load_model, probabilities_for_context

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
_IDX_TO_BASE = ("A", "C", "G", "T")
PHRED_MIN = 0
PHRED_MAX = 60
N_PHRED = PHRED_MAX - PHRED_MIN + 1
ERROR_TYPE_NAMES = (
    "match",
    "sub_to_A",
    "sub_to_C",
    "sub_to_G",
    "sub_to_T",
    "ins_A",
    "ins_C",
    "ins_G",
    "ins_T",
    "deletion",
)
PHRED_BINS = (
    (0, 9, "0-9"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, 60, "40-60"),
)
DEFAULT_WIGGLE_WINDOW_BP = 50


def _load_artifact_metadata(path: Path) -> tuple[str, str, int]:
    """Return model identifier, parameterisation, and context length."""
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")

    context_length = int(artifact["context_length"])
    parameterization = str(
        artifact.get("parameterization", artifact.get("model_type", ""))
    )
    if parameterization not in {"additive_context", "combinatorial_context"}:
        params = artifact["maximum_likelihood"]["params"]
        parameterization = (
            "additive_context" if "intercept_logits" in params else "combinatorial_context"
        )
    model_id = str(artifact.get("model_id", f"recovery_context_{context_length}"))
    return model_id, parameterization, context_length


def _skiver_command(skiver_bin: Path) -> list[str]:
    """Return the command prefix used to invoke skiver."""
    if skiver_bin.exists():
        return [str(skiver_bin)]
    return ["cargo", "run", "--quiet", "--"]


def _run(command: Sequence[str], *, cwd: Path) -> None:
    """Run a subprocess and raise on failure."""
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _genome_blender_command(
    *,
    genome_blender_dir: Path,
    conda_env: str | None,
    python_executable: Path | None,
) -> list[str]:
    """Return the command prefix used to invoke genome-blender."""
    generate_reads = genome_blender_dir / "generate_reads.py"
    if python_executable is not None:
        return [str(python_executable), str(generate_reads)]
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python", str(generate_reads)]
    return [sys.executable, str(generate_reads)]


def _write_genome_blender_input(path: Path, reference: Path) -> None:
    """Write a one-genome abundance table for genome-blender."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["genome_id", "fasta_path", "abundance"])
        writer.writerow(["synthetic", str(reference), "1.0"])


def _simulate_with_genome_blender(
    *,
    model: Path,
    reference: Path,
    output_prefix: Path,
    input_csv: Path,
    num_reads: int,
    seed: int,
    joint_phred_calibration: Path | None,
    quality_calibration_model: dict[str, object] | None,
    use_vi: bool,
    genome_blender_cmd: Sequence[str],
) -> Path:
    """Run genome-blender for one split and return the FASTQ path."""
    _write_genome_blender_input(input_csv, reference)
    command = [
        *genome_blender_cmd,
        "--input-csv",
        str(input_csv),
        "--num-reads",
        str(num_reads),
        "--output-prefix",
        str(output_prefix),
        "--single-end",
        "--amplicon",
        "--long-read",
        "--skiver-error-model",
        str(model),
        "--seed",
        str(seed),
        "--no-compress",
        "--no-ansi",
    ]
    if joint_phred_calibration is not None:
        command.extend(["--skiver-phred-calibration", str(joint_phred_calibration)])
    if quality_calibration_model is not None:
        command.extend(_quality_calibration_cli_args(quality_calibration_model))
    if use_vi:
        command.append("--skiver-use-vi")
    _run(command, cwd=REPO_ROOT)
    return output_prefix.with_suffix(".fastq")


def _load_quality_calibration_model(path: Path | None) -> dict[str, object] | None:
    """Load a genome-blender Q-to-error calibration artifact."""
    if path is None:
        return None
    with open(path) as handle:
        artifact = json.load(handle)
    config = artifact.get("genome_blender_config", artifact)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a genome_blender_config object")
    return config


def _quality_calibration_cli_args(config: dict[str, object]) -> list[str]:
    """Return genome-blender CLI arguments for a calibration config."""
    model = str(config["quality_calibration_model"])
    args = ["--quality-calibration-model", model]
    option_map = {
        "qcal_intercept": "--qcal-intercept",
        "qcal_slope": "--qcal-slope",
        "qcal_floor": "--qcal-floor",
        "qcal_ceiling": "--qcal-ceiling",
        "qcal_steepness": "--qcal-steepness",
        "qcal_midpoint": "--qcal-midpoint",
    }
    for key, option in option_map.items():
        if key in config:
            args.extend([option, str(config[key])])
    return args


def _dump(
    *,
    skiver_cmd: Sequence[str],
    reads: Sequence[Path],
    prefix: Path,
    k: int,
    v: int,
    c: int,
    forward_only: bool,
) -> None:
    """Run `skiver dump --base --raw` for one simulated split."""
    command = [
        *skiver_cmd,
        "dump",
        *[str(read) for read in reads],
        "-o",
        str(prefix),
        "--base",
        "--raw",
        "--use-all",
        "-k",
        str(k),
        "-v",
        str(v),
        "-c",
        str(c),
    ]
    if forward_only:
        command.append("--forward-only")
    _run(command, cwd=REPO_ROOT)


def _analyze_weibull_rate(
    *,
    skiver_cmd: Sequence[str],
    reads: Sequence[Path],
    prefix: Path,
    k: int,
    v: int,
    c: int,
    forward_only: bool,
    weibull_outlier_threshold: float,
) -> dict[str, object] | None:
    """Run `skiver analyze` and return Weibull parameters for the training reads.

    Uses (1 - exp(-λ·v^β)) / v, the mean per-base survival probability over a
    v-length window, rather than the position-1 hazard 1 - exp(-λ).

    The outlier filter is applied before the Weibull fit (skiver's iterative
    Binomial test against the fitted hazard). The threshold controls how
    aggressively high-error keys are removed. The default 1e-3 is approximately
    Bonferroni-corrected at p<0.05 for typical key counts (~50-100 per amplicon
    dataset), and empirically removes read-end-clustered keys that would otherwise
    inflate λ without removing legitimate high-error contexts.

    Returns None if the output CSV cannot be parsed (e.g. too little data).
    Returns a dict with keys: ``rate`` (window-averaged per-base error rate),
    ``lambda_mean``, ``beta_mean`` (means over keys), ``lambda_values`` and
    ``beta_values`` (per-key lists).
    """
    import csv
    import math

    command = [
        *skiver_cmd,
        "analyze",
        *[str(read) for read in reads],
        "-o",
        str(prefix),
        "-k",
        str(k),
        "-v",
        str(v),
        "-c",
        str(c),
        "-e",
        str(weibull_outlier_threshold),
    ]
    if forward_only:
        command.append("--forward-only")
    _run(command, cwd=REPO_ROOT)
    csv_path = prefix.parent / f"{prefix.name}.summary_error_rate.csv"
    if not csv_path.exists():
        logger.warning("skiver analyze did not produce %s", csv_path)
        return None
    with open(csv_path) as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("lambda") and row.get("beta")]
    if not rows:
        logger.warning("No lambda/beta rows in %s", csv_path)
        return None
    lambdas = [float(r["lambda"]) for r in rows]
    betas = [float(r["beta"]) for r in rows]
    rates = [
        (1.0 - math.exp(-lam * v ** beta)) / v
        for lam, beta in zip(lambdas, betas)
    ]
    rate = float(np.mean(rates))
    lambda_mean = float(np.mean(lambdas))
    beta_mean = float(np.mean(betas))
    logger.info(
        "Weibull window-averaged per-base error rate (v=%d): %.6f  λ=%.4f  β=%.4f",
        v, rate, lambda_mean, beta_mean,
    )
    return {
        "rate": rate,
        "lambda_mean": lambda_mean,
        "beta_mean": beta_mean,
        "lambda_values": lambdas,
        "beta_values": betas,
    }


def _counts_for_training(
    *,
    train_prefix: Path,
    test_prefix: Path,
    context_length: int,
    additive_context: bool,
    max_contexts: int | None,
) -> tuple[ContextCounts, ContextCounts]:
    """Aggregate skiver dump rows and return train/test count tensors."""
    train_screen = aggregate_context_length_screen_counts(
        [train_prefix],
        context_lengths=[context_length],
        passes_filter_only=False,
    )
    test_screen = aggregate_context_length_screen_counts(
        [test_prefix],
        context_lengths=[context_length],
        passes_filter_only=False,
    )
    train_counts = _screen_counts(
        train_screen,
        context_length=context_length,
        additive_context=additive_context,
    )
    test_counts = _screen_counts(
        test_screen,
        context_length=context_length,
        additive_context=additive_context,
    )
    if additive_context and max_contexts is not None:
        train_counts = subsample_context_counts(train_counts, max_contexts)
    return train_counts, test_counts


def _screen_counts(
    screen: ContextLengthScreenCounts,
    *,
    context_length: int,
    additive_context: bool,
) -> ContextCounts:
    """Return one context-count object with the requested parameterisation."""
    counts = screen.by_length[context_length]
    return dataclasses.replace(counts, additive_context=additive_context)


def _save_retrained_artifact(
    path: Path,
    *,
    model_id: str,
    parameterization: str,
    context_length: int,
    fit_params: dict[str, torch.Tensor],
    fit_losses: Sequence[float],
    train_log_likelihood: float,
    test_log_likelihood: float,
    vi_fit: object | None,
    train_counts: ContextCounts,
    test_counts: ContextCounts,
    v: int = 13,
    weibull_rate: float | None = None,
) -> None:
    """Save a minimal artifact compatible with Skiver context simulators."""
    maximum_likelihood = {
        "params": fit_params,
        "losses": torch.tensor(fit_losses),
        "train_log_likelihood": train_log_likelihood,
        "test_log_likelihood": test_log_likelihood,
    }
    artifact = {
        "model_id": f"{model_id}_synthetic_retrained",
        "model_type": parameterization,
        "parameterization": parameterization,
        "platform": "synthetic",
        "context_length": context_length,
        "target": "error_type",
        "n_train": train_counts.total_observations,
        "n_test": test_counts.total_observations,
        "maximum_likelihood": maximum_likelihood,
    }
    if vi_fit is not None:
        artifact["variational_inference"] = {
            "params_mean": vi_fit.params_mean,
            "params_stdev": vi_fit.params_stdev,
            "inference_params": vi_fit.inference_params,
            "losses": torch.tensor(vi_fit.losses),
            "train_log_likelihood": vi_fit.train_log_likelihood,
            "test_log_likelihood": vi_fit.test_log_likelihood,
            "train_elbo": vi_fit.train_elbo,
            "test_elbo": vi_fit.test_elbo,
            "prior_scale": vi_fit.prior_scale,
        }
    calibration: dict[str, float] | None = None
    if weibull_rate is not None:
        additive = parameterization == "additive_context"
        raw_mle = compute_marginal_error_rate(
            train_counts.counts, fit_params,
            run_values=train_counts.run_values,
            additive_context=additive,
            context_indices=train_counts.context_indices,
        )
        delta_mle = calibrate_to_rate(
            train_counts.counts, fit_params, weibull_rate,
            run_values=train_counts.run_values,
            additive_context=additive,
            context_indices=train_counts.context_indices,
        )
        calibration = {
            "weibull_target_rate": weibull_rate,
            "uncalibrated_mle_rate": raw_mle,
            "calibration_offset_mle": delta_mle,
        }
        if vi_fit is not None:
            raw_vi = compute_marginal_error_rate(
                train_counts.counts, vi_fit.params_mean,
                run_values=train_counts.run_values,
                additive_context=additive,
                context_indices=train_counts.context_indices,
            )
            delta_vi = calibrate_to_rate(
                train_counts.counts, vi_fit.params_mean, weibull_rate,
                run_values=train_counts.run_values,
                additive_context=additive,
                context_indices=train_counts.context_indices,
            )
            calibration["uncalibrated_vi_rate"] = raw_vi
            calibration["calibration_offset_vi"] = delta_vi
        logger.info(
            "Weibull calibration: MLE %.6f → %.6f (δ=%+.4f)",
            raw_mle, weibull_rate, delta_mle,
        )
    artifact["calibration"] = calibration

    # Marginal Weibull parameters predicted by the model on the training distribution.
    # Uses a mixture-of-geometrics approximation; see compute_marginal_weibull docstring.
    mle_weibull = compute_marginal_weibull(
        train_counts.counts, fit_params, v,
        run_values=train_counts.run_values,
        additive_context=additive,
        context_indices=train_counts.context_indices,
    )
    marginal_weibull: dict[str, object] = {"maximum_likelihood": mle_weibull}
    if vi_fit is not None:
        vi_weibull = compute_marginal_weibull(
            train_counts.counts, vi_fit.params_mean, v,
            run_values=train_counts.run_values,
            additive_context=additive,
            context_indices=train_counts.context_indices,
        )
        marginal_weibull["variational_inference"] = vi_weibull
    artifact["marginal_weibull"] = marginal_weibull
    artifact["v"] = v

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def _load_artifact(path: Path) -> dict[str, object]:
    """Load a torch artifact with backwards-compatible options."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Return compact numeric stats for a tensor."""
    flat = tensor.detach().cpu().float().reshape(-1)
    if flat.numel() == 0:
        return {"n": 0}
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _compare_tensors(source: torch.Tensor, retrained: torch.Tensor) -> dict[str, float]:
    """Return comparison stats for two equal-shaped tensors."""
    s = source.detach().cpu().float().reshape(-1)
    r = retrained.detach().cpu().float().reshape(-1)
    if s.shape != r.shape:
        return {
            "source_n": int(s.numel()),
            "retrained_n": int(r.numel()),
            "shape_match": False,
        }
    diff = r - s
    if s.numel() > 1 and float(s.std().item()) > 0 and float(r.std().item()) > 0:
        corr = float(torch.corrcoef(torch.stack([s, r]))[0, 1].item())
    else:
        corr = float("nan")
    return {
        "n": int(s.numel()),
        "shape_match": True,
        "source_mean": float(s.mean().item()),
        "retrained_mean": float(r.mean().item()),
        "mean_ratio": float((r.mean() / s.mean()).item()) if float(s.mean()) != 0 else float("nan"),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "pearson": corr,
    }


def _vi_uncertainty_metrics(source_model: Path, retrained_model: Path) -> dict[str, object]:
    """Compare source and retrained VI posterior standard deviations."""
    source_artifact = _load_artifact(source_model)
    retrained_artifact = _load_artifact(retrained_model)
    source_vi = source_artifact.get("variational_inference")
    retrained_vi = retrained_artifact.get("variational_inference")
    if not isinstance(source_vi, dict) or not isinstance(retrained_vi, dict):
        return {
            "available": False,
            "reason": "source or retrained artifact lacks variational_inference",
        }

    source_stdev = source_vi.get("params_stdev", {})
    retrained_stdev = retrained_vi.get("params_stdev", {})
    if not isinstance(source_stdev, dict) or not isinstance(retrained_stdev, dict):
        return {"available": False, "reason": "missing params_stdev dictionaries"}

    shared = sorted(set(source_stdev).intersection(retrained_stdev))
    per_parameter = {
        name: _compare_tensors(source_stdev[name], retrained_stdev[name])
        for name in shared
    }
    return {
        "available": True,
        "source_model": str(source_model),
        "retrained_model": str(retrained_model),
        "source_prior_scale": source_vi.get("prior_scale"),
        "retrained_prior_scale": retrained_vi.get("prior_scale"),
        "source_parameters": sorted(source_stdev),
        "retrained_parameters": sorted(retrained_stdev),
        "shared_parameters": shared,
        "source_stdev_summary": {
            name: _tensor_stats(source_stdev[name]) for name in sorted(source_stdev)
        },
        "retrained_stdev_summary": {
            name: _tensor_stats(retrained_stdev[name]) for name in sorted(retrained_stdev)
        },
        "per_parameter": per_parameter,
    }


def _context_index(history: Sequence[str], context_length: int) -> int:
    """Encode the last `context_length` bases as a base-4 context index."""
    context_index = 0
    for base in history[-context_length:]:
        context_index = context_index * 4 + BASE_TO_IDX[base]
    return context_index


def _raw_key_contexts_for_obs_ids(
    base_observations: Path,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (obs_ids, context_codes) from raw dump as compact sorted int64 arrays.

    Returns empty arrays when context_length <= 0 or the raw file is absent.
    Each context_code encodes context_length bases as a base-4 integer (MSB = leftmost base).
    """
    raw_path = base_observations.with_name(
        base_observations.name.replace(".base_observations.tsv", ".raw_observations.tsv")
    )
    _empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if context_length <= 0 or not raw_path.exists():
        return _empty

    obs_id_list: list[int] = []
    code_list: list[int] = []
    with open(raw_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "obs_id" not in (reader.fieldnames or []) or "key_str" not in (
            reader.fieldnames or []
        ):
            return _empty
        for row in reader:
            key = row["key_str"].upper()
            if len(key) < context_length or any(base not in BASE_TO_IDX for base in key):
                continue
            code = 0
            for base in key[-context_length:]:
                code = code * 4 + BASE_TO_IDX[base]
            obs_id_list.append(int(row["obs_id"]))
            code_list.append(code)

    if not obs_id_list:
        return _empty

    obs_ids_arr = np.array(obs_id_list, dtype=np.int64)
    codes_arr = np.array(code_list, dtype=np.int64)
    order = np.argsort(obs_ids_arr, kind="stable")
    return obs_ids_arr[order], codes_arr[order]


def _observed_records_from_dump(
    base_observations: Path,
    *,
    context_length: int,
) -> list[tuple[int, int, int, int]]:
    """Load sparse observed rows as (context, true_base, error_type, count).

    The old in-repo simulator emitted an explicit truth JSON.  genome-blender is
    now the simulator, so the benchmark compares source/retrained probabilities
    over the context/base exposure seen in Skiver's `dump --base` output.
    """

    raw_obs_ids, raw_codes = _raw_key_contexts_for_obs_ids(base_observations, context_length)

    def _add_observation(rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        obs_id = int(rows[0]["obs_id"])
        if raw_obs_ids.size:
            idx = int(np.searchsorted(raw_obs_ids, obs_id))
            if idx < raw_obs_ids.size and raw_obs_ids[idx] == obs_id:
                code = int(raw_codes[idx])
                history = [
                    _IDX_TO_BASE[(code >> (2 * (context_length - 1 - i))) & 3]
                    for i in range(context_length)
                ]
            else:
                history = []
        else:
            history = []
        if not history:
            prev_base = rows[0]["prev_base"]
            history = [prev_base] if prev_base in BASE_TO_IDX else []

        for index, row in enumerate(rows):
            true_base = row["true_base"]
            obs_base = row["obs_base"]
            event_true_base = true_base
            if true_base == "-":
                for next_row in rows[index + 1:]:
                    if next_row["true_base"] in BASE_TO_IDX:
                        event_true_base = next_row["true_base"]
                        break

            if event_true_base in BASE_TO_IDX and len(history) >= context_length:
                context = _context_index(history, context_length)
                error_type = encode_error_type(true_base, obs_base, row["edit_op"])
                counts[(context, BASE_TO_IDX[event_true_base], error_type)] += 1

            if true_base in BASE_TO_IDX:
                history.append(true_base)

    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    current_obs_id: int | None = None
    current_rows: list[dict[str, str]] = []
    with open(base_observations, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            obs_id = int(row["obs_id"])
            if obs_id != current_obs_id:
                _add_observation(current_rows)
                current_obs_id = obs_id
                current_rows = []
            current_rows.append(row)
    _add_observation(current_rows)

    return [
        (context, true_base, error_type, count)
        for (context, true_base, error_type), count in sorted(counts.items())
    ]


def _observed_counts(records: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Return empirical event counts by error type."""
    counts = np.zeros(NUM_ERROR_TYPES, dtype=np.float64)
    for _, _, error_type, count in records:
        counts[error_type] += count
    return counts


def _phred_bin_label(phred: int) -> str:
    """Return a compact display bin for a Phred score."""
    if phred < 0:
        return "deletion_or_missing"
    for low, high, label in PHRED_BINS:
        if low <= phred <= high:
            return label
    return f">{PHRED_BINS[-1][1]}"


def _normalise_fastq_name(name: str) -> str:
    """Return the BAM query name component of a FASTQ record name."""
    value = name[1:] if name.startswith("@") else name
    value = value.split()[0]
    if value.endswith("/1") or value.endswith("/2"):
        value = value[:-2]
    return value


def _mate_from_fastq_name(name: str, path: Path, fallback: str) -> str:
    """Return R1/R2 for a FASTQ record or path."""
    token = name[1:] if name.startswith("@") else name
    token = token.split()[0]
    if token.endswith("/1"):
        return "R1"
    if token.endswith("/2"):
        return "R2"
    stem = path.name
    if "_R1" in stem or ".R1" in stem:
        return "R1"
    if "_R2" in stem or ".R2" in stem:
        return "R2"
    return fallback


def _fastq_read_index(reads: Sequence[Path]) -> list[dict[str, object]]:
    """Return read-id indexed FASTQ metadata in Skiver input order."""
    index: list[dict[str, object]] = []
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    for file_idx, path in enumerate(reads):
        fallback = "R1" if file_idx == 0 else f"R{file_idx + 1}"
        record_idx = 0
        with open(path) as handle:
            while True:
                header = handle.readline().strip()
                if not header:
                    break
                sequence = handle.readline().strip()
                handle.readline()
                handle.readline()
                query_name = _normalise_fastq_name(header)
                mate = _mate_from_fastq_name(header, path, fallback)
                name_mate_key = (query_name, mate)
                occurrence_index = occurrence_counts[name_mate_key]
                occurrence_counts[name_mate_key] += 1
                index.append(
                    {
                        "fastq": str(path),
                        "record_index": record_idx,
                        "mate": mate,
                        "query_name": query_name,
                        "occurrence_index": occurrence_index,
                        "read_length": len(sequence),
                    }
                )
                record_idx += 1
    return index


def _bam_query_to_fastq_pos(
    query_pos: int,
    read_length: int,
    is_reverse: bool,
) -> int:
    """Convert a BAM query position to the original FASTQ coordinate."""
    if is_reverse:
        return read_length - 1 - query_pos
    return query_pos


def _skiver_row_physical_pos(
    *,
    is_forward: bool,
    read_pos: int,
    t: int,
) -> int:
    """Return the physical FASTQ coordinate represented by a Skiver base row."""
    if is_forward:
        return read_pos
    return read_pos - (2 * t) + 1


@functools.lru_cache(maxsize=None)
def _load_fasta(path: Path) -> dict[str, str]:
    """Load FASTA records into an ID-to-sequence mapping.

    Memoised on ``path``: the reference is parsed once and the same dict is
    returned to every caller (``_physical_truth_from_bam`` and
    ``_truth_maps_from_bam`` each load it per split). The combined metagenome
    reference is hundreds of MB, so reloading it 4-6× per run was a large,
    avoidable transient. Callers must treat the returned dict as read-only.
    """
    records: dict[str, str] = {}
    current_name: str | None = None
    current_parts: list[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records[current_name] = "".join(current_parts).upper()
                current_name = line[1:].split()[0]
                current_parts = []
            else:
                current_parts.append(line)
    if current_name is not None:
        records[current_name] = "".join(current_parts).upper()
    return records


def _reference_sequence_for_bam_name(
    reference_name: str,
    references: dict[str, str],
) -> str:
    """Return the reference sequence for a BAM reference name."""
    normalised_name = _normalise_bam_reference_name(reference_name, references)
    if normalised_name in references:
        return references[normalised_name]
    raise KeyError(reference_name)


def _normalise_bam_reference_name(
    reference_name: str,
    references: dict[str, str],
) -> str:
    """Return the FASTA record ID represented by a BAM reference name."""
    if reference_name in references:
        return reference_name
    if ":" in reference_name:
        suffix = reference_name.split(":", 1)[1]
        if suffix in references:
            return suffix
    raise KeyError(reference_name)


def _nearest_ref_pos_from_aligned_pairs(
    pairs: Sequence[tuple[int | None, int | None]],
    index: int,
) -> int | None:
    """Return a nearby reference position for an insertion-only pair."""
    for prev_index in range(index - 1, -1, -1):
        ref_pos = pairs[prev_index][1]
        if ref_pos is not None:
            return int(ref_pos)
    for next_index in range(index + 1, len(pairs)):
        ref_pos = pairs[next_index][1]
        if ref_pos is not None:
            return int(ref_pos)
    return None


def _nearest_distance(sorted_positions: Sequence[int], position: int | None) -> int | None:
    """Return distance to the nearest sorted position, or None if unavailable."""
    if position is None or not sorted_positions:
        return None
    idx = bisect.bisect_left(sorted_positions, position)
    candidates: list[int] = []
    if idx < len(sorted_positions):
        candidates.append(abs(sorted_positions[idx] - position))
    if idx > 0:
        candidates.append(abs(sorted_positions[idx - 1] - position))
    return min(candidates) if candidates else None


def _q_counts_dict(q_counts: np.ndarray) -> dict[str, list[int]]:
    """Return sparse Q-count lists keyed by error-type name."""
    result: dict[str, list[int]] = {}
    for idx, name in enumerate(ERROR_TYPE_NAMES):
        row = q_counts[idx]
        if int(row.sum()) > 0:
            result[name] = row.astype(int).tolist()
    return result


def _physical_truth_from_bam(
    bam_path: Path,
    reference: Path,
) -> dict[str, object]:
    """Return physical base-level truth counts from a genome-blender BAM."""
    if not bam_path.exists():
        return {"available": False, "reason": f"Missing BAM: {bam_path}"}
    try:
        import pysam  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"available": False, "reason": f"pysam unavailable: {exc}"}

    references = _load_fasta(reference)
    counts = np.zeros(NUM_ERROR_TYPES, dtype=np.int64)
    q_counts = np.zeros((NUM_ERROR_TYPES, N_PHRED), dtype=np.int64)
    total_records = 0
    aligned_pairs = 0
    reference_bases = 0
    query_bases = 0
    op_counts: Counter[str] = Counter()

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            total_records += 1
            ref_seq = _reference_sequence_for_bam_name(read.reference_name, references)
            query = read.query_sequence or ""
            qualities = read.query_qualities
            for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
                if query_pos is None and ref_pos is not None:
                    counts[NUM_ERROR_TYPES - 1] += 1
                    op_counts["deletion"] += 1
                    reference_bases += 1
                    continue
                if ref_pos is None and query_pos is not None:
                    obs_base = query[query_pos].upper()
                    error_type = 5 + BASE_TO_IDX.get(obs_base, 0)
                    counts[error_type] += 1
                    op_counts[f"ins_{obs_base}"] += 1
                    query_bases += 1
                    if qualities is not None:
                        q = min(max(int(qualities[query_pos]), PHRED_MIN), PHRED_MAX)
                        q_counts[error_type, q] += 1
                    continue
                if query_pos is None or ref_pos is None:
                    continue

                true_base = ref_seq[ref_pos].upper()
                obs_base = query[query_pos].upper()
                aligned_pairs += 1
                reference_bases += 1
                query_bases += 1
                if true_base == obs_base:
                    error_type = 0
                    op_counts["match"] += 1
                else:
                    error_type = 1 + BASE_TO_IDX.get(obs_base, 0)
                    op_counts[f"{true_base}>{obs_base}"] += 1
                counts[error_type] += 1
                if qualities is not None:
                    q = min(max(int(qualities[query_pos]), PHRED_MIN), PHRED_MAX)
                    q_counts[error_type, q] += 1

    return {
        "available": True,
        "bam": str(bam_path),
        "n_reads": total_records,
        "aligned_pairs": aligned_pairs,
        "reference_bases": reference_bases,
        "query_bases": query_bases,
        "counts": counts.astype(int).tolist(),
        "probs": _normalise(counts).tolist(),
        "error_rate": _error_rate(counts),
        "q_counts_by_error_type": _q_counts_dict(q_counts),
        "top_operations": [
            {"operation": op, "count": int(count)}
            for op, count in op_counts.most_common(20)
        ],
    }


def _truth_maps_from_bam(
    bam_path: Path,
    reference: Path,
) -> dict[str, object]:
    """Return BAM truth events keyed by query name and mate."""
    if not bam_path.exists():
        return {"available": False, "reason": f"Missing BAM: {bam_path}"}
    try:
        import pysam  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"available": False, "reason": f"pysam unavailable: {exc}"}

    references = _load_fasta(reference)
    reads: dict[tuple[str, str, int], dict[str, object]] = {}
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    coordinate_truth_errors = 0
    coordinate_truth_by_type: Counter[str] = Counter()
    deletion_truth_errors = 0
    duplicate_name_mate_keys: set[tuple[str, str]] = set()
    truth_positions_by_reference: defaultdict[str, list[int]] = defaultdict(list)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if read.reference_name is None:
                continue
            mate = "R2" if read.is_read2 else "R1"
            read_length = int(read.query_length or 0)
            name_mate_key = (read.query_name, mate)
            occurrence_index = occurrence_counts[name_mate_key]
            if occurrence_index > 0:
                duplicate_name_mate_keys.add(name_mate_key)
            occurrence_counts[name_mate_key] += 1
            read_key = (read.query_name, mate, occurrence_index)
            read_events: defaultdict[int, list[str]] = defaultdict(list)
            read_error_types: defaultdict[int, list[str]] = defaultdict(list)
            deletion_ref_positions: list[int] = []
            reference_name = _normalise_bam_reference_name(read.reference_name, references)
            ref_seq = references[reference_name]
            ref_pos_by_fastq_pos: dict[int, int] = {}
            query = read.query_sequence or ""
            pairs = list(read.get_aligned_pairs(matches_only=False))
            for pair_index, (query_pos, ref_pos) in enumerate(pairs):
                if query_pos is None and ref_pos is not None:
                    deletion_truth_errors += 1
                    deletion_ref_positions.append(int(ref_pos))
                    continue
                if query_pos is None:
                    continue

                fastq_pos = _bam_query_to_fastq_pos(
                    int(query_pos),
                    read_length,
                    read.is_reverse,
                )
                if ref_pos is None:
                    event = "insertion"
                    approximate_ref_pos = _nearest_ref_pos_from_aligned_pairs(
                        pairs,
                        pair_index,
                    )
                    obs_base = query[int(query_pos)].upper()
                    coordinate_truth_errors += 1
                    coordinate_truth_by_type[event] += 1
                    read_events[fastq_pos].append(event)
                    read_error_types[fastq_pos].append(f"ins_{obs_base}")
                    if approximate_ref_pos is not None:
                        truth_positions_by_reference[reference_name].append(
                            approximate_ref_pos
                        )
                    continue

                ref_pos_by_fastq_pos[fastq_pos] = int(ref_pos)
                true_base = ref_seq[int(ref_pos)].upper()
                obs_base = query[int(query_pos)].upper()
                if true_base == obs_base:
                    read_events[fastq_pos].append("match")
                else:
                    event = "substitution"
                    coordinate_truth_errors += 1
                    coordinate_truth_by_type[event] += 1
                    read_events[fastq_pos].append(event)
                    read_error_types[fastq_pos].append(f"{true_base}>{obs_base}")
                    truth_positions_by_reference[reference_name].append(int(ref_pos))

            reads[read_key] = {
                "length": read_length,
                "reference_name": reference_name,
                "ref_pos_by_fastq_pos": ref_pos_by_fastq_pos,
                "events_by_fastq_pos": {
                    int(pos): list(events) for pos, events in read_events.items()
                },
                "error_types_by_fastq_pos": {
                    int(pos): list(events)
                    for pos, events in read_error_types.items()
                },
                "deletion_ref_positions": deletion_ref_positions,
            }

    return {
        "available": True,
        "bam": str(bam_path),
        "reads": reads,
        "coordinate_truth_errors": coordinate_truth_errors,
        "coordinate_truth_by_type": {
            key: int(value) for key, value in coordinate_truth_by_type.items()
        },
        "duplicate_name_mate_keys": len(duplicate_name_mate_keys),
        "max_name_mate_occurrences": max(occurrence_counts.values(), default=0),
        "truth_positions_by_reference": {
            key: sorted(set(value))
            for key, value in truth_positions_by_reference.items()
        },
        "deletion_truth_errors": deletion_truth_errors,
    }


def _read_truth_for_meta(
    truth_maps: dict[str, object],
    read_meta: dict[str, object],
) -> dict[str, object] | None:
    """Return BAM truth for one FASTQ read metadata row."""
    reads = truth_maps.get("reads", {})
    key = (
        read_meta["query_name"],
        read_meta["mate"],
        int(read_meta.get("occurrence_index", 0)),
    )
    return reads.get(key)  # type: ignore[union-attr]


def _truth_classes_at_pos(
    truth_maps: dict[str, object],
    read_meta: dict[str, object],
    physical_pos: int,
) -> tuple[list[str], list[str]]:
    """Return truth classes and concrete truth events for one FASTQ coordinate."""
    read_truth = _read_truth_for_meta(truth_maps, read_meta)
    if read_truth is None:
        return ["unmapped_read"], []
    events_by_pos = read_truth["events_by_fastq_pos"]  # type: ignore[index]
    error_types_by_pos = read_truth["error_types_by_fastq_pos"]  # type: ignore[index]
    events = list(events_by_pos.get(physical_pos, []))
    if not events:
        return ["unmapped_position"], []
    truth_events = list(error_types_by_pos.get(physical_pos, []))
    if any(event != "match" for event in events):
        return [event for event in events if event != "match"], truth_events
    return ["match"], []


def _increment_nested(counter: Counter[tuple[str, str]], left: str, right: str) -> None:
    """Increment a two-dimensional counter."""
    counter[(left, right)] += 1


def _nested_counter_rows(
    counter: Counter[tuple[str, str]],
    left_name: str,
    right_name: str,
) -> list[dict[str, object]]:
    """Return JSON rows from a two-dimensional counter."""
    return [
        {left_name: left, right_name: right, "count": int(count)}
        for (left, right), count in sorted(counter.items())
    ]


def _skiver_bam_error_match_split(
    *,
    split: str,
    base_observations: Path,
    reads: Sequence[Path],
    bam_path: Path,
    reference: Path,
    wiggle_window_bp: int = DEFAULT_WIGGLE_WINDOW_BP,
) -> dict[str, object]:
    """Compare Skiver-detected base rows with BAM truth for one split."""
    if not base_observations.exists():
        return {
            "split": split,
            "available": False,
            "reason": f"Missing base observations: {base_observations}",
        }
    truth_maps = _truth_maps_from_bam(bam_path, reference)
    if not truth_maps.get("available"):
        return {
            "split": split,
            "available": False,
            "reason": truth_maps.get("reason", "BAM truth unavailable"),
        }

    read_index = _fastq_read_index(reads)
    total_rows = 0
    detected_errors = 0
    same_coordinate_true = 0
    same_coordinate_false = 0
    same_sequence_true = 0
    same_sequence_false = 0
    wiggle_true = 0
    wiggle_false = 0
    unmapped_detected = 0
    invalid_coordinate_rows = 0
    obs_ids_with_truth: set[int] = set()
    detected_obs_ids: list[int] = []
    detected_truth_coordinates: Counter[tuple[int, int]] = Counter()
    detected_truth_reference_positions: Counter[tuple[str, int]] = Counter()
    nearest_truth_distances: list[int] = []
    detected_by_skiver_truth: Counter[tuple[str, str]] = Counter()
    detected_by_skiver_same_sequence: Counter[tuple[str, str]] = Counter()
    detected_by_skiver_wiggle: Counter[tuple[str, str]] = Counter()
    detected_by_mate_truth: Counter[tuple[str, str]] = Counter()
    detected_by_forward_truth: Counter[tuple[str, str]] = Counter()
    detected_by_phred_truth: Counter[tuple[str, str]] = Counter()
    truth_positions_by_reference = truth_maps.get("truth_positions_by_reference", {})
    truth_coordinate_keys: set[tuple[int, int]] = set()
    comparable_fastq_coordinates = 0
    for read_id, read_meta in enumerate(read_index):
        read_truth = _read_truth_for_meta(truth_maps, read_meta)
        if read_truth is None:
            continue
        comparable_fastq_coordinates += int(read_meta["read_length"])
        events_by_pos = read_truth["events_by_fastq_pos"]  # type: ignore[index]
        for pos, events in events_by_pos.items():  # type: ignore[union-attr]
            if any(event in {"substitution", "insertion"} for event in events):
                truth_coordinate_keys.add((read_id, int(pos)))
    detected_coordinate_keys: set[tuple[int, int]] = set()
    skiver_covered_coordinate_keys: set[tuple[int, int]] = set()

    with open(base_observations, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_rows += 1
            try:
                read_id = int(row["read_id"])
                read_pos = int(row["read_pos"])
                t = int(row["t"])
            except (KeyError, ValueError):
                invalid_coordinate_rows += 1
                continue
            if read_id < 0 or read_id >= len(read_index):
                if row.get("edit_op") != "NA":
                    detected_errors += 1
                    unmapped_detected += 1
                continue

            read_meta = read_index[read_id]
            read_truth = _read_truth_for_meta(truth_maps, read_meta)
            physical_pos = _skiver_row_physical_pos(
                is_forward=row.get("is_forward") == "true",
                read_pos=read_pos,
                t=t,
            )
            read_length = int(read_meta["read_length"])
            valid_coordinate = 0 <= physical_pos < read_length
            if not valid_coordinate:
                invalid_coordinate_rows += 1
            elif read_truth is not None:
                skiver_covered_coordinate_keys.add((read_id, physical_pos))

            truth_classes: list[str]
            truth_events: list[str]
            if valid_coordinate:
                truth_classes, truth_events = _truth_classes_at_pos(
                    truth_maps,
                    read_meta,
                    physical_pos,
                )
                if any(truth_class in {"substitution", "insertion"} for truth_class in truth_classes):
                    obs_ids_with_truth.add(int(row["obs_id"]))
            else:
                truth_classes, truth_events = ["invalid_coordinate"], []

            skiver_error_type = encode_error_type(
                row.get("true_base", "N"),
                row.get("obs_base", "N"),
                row.get("edit_op", "NA"),
            )
            if skiver_error_type == 0:
                continue

            detected_errors += 1
            obs_id = int(row["obs_id"])
            detected_obs_ids.append(obs_id)
            if valid_coordinate and read_truth is not None:
                detected_coordinate_keys.add((read_id, physical_pos))
            truth_label = (
                "+".join(sorted(set(truth_classes)))
                if truth_classes
                else "unmapped_position"
            )
            if any(truth_class in {"substitution", "insertion"} for truth_class in truth_classes):
                same_coordinate_true += 1
                detected_truth_coordinates[(read_id, physical_pos)] += 1
            elif truth_label in {"invalid_coordinate", "unmapped_read", "unmapped_position"}:
                unmapped_detected += 1
            else:
                same_coordinate_false += 1

            skiver_ref_name: str | None = None
            skiver_ref_pos: int | None = None
            if read_truth is not None and valid_coordinate:
                skiver_ref_name = str(read_truth.get("reference_name"))
                ref_pos_map = read_truth.get("ref_pos_by_fastq_pos", {})
                if physical_pos in ref_pos_map:  # type: ignore[operator]
                    skiver_ref_pos = int(ref_pos_map[physical_pos])  # type: ignore[index]

            ref_truth_positions = []
            if skiver_ref_name is not None:
                ref_truth_positions = list(
                    truth_positions_by_reference.get(skiver_ref_name, [])  # type: ignore[union-attr]
                )

            if ref_truth_positions:
                same_sequence_true += 1
                same_sequence_label = "same_sequence_true"
            else:
                same_sequence_false += 1
                same_sequence_label = "same_sequence_false"

            nearest_distance = _nearest_distance(ref_truth_positions, skiver_ref_pos)
            if nearest_distance is not None:
                nearest_truth_distances.append(nearest_distance)
            if nearest_distance is not None and nearest_distance <= wiggle_window_bp:
                wiggle_true += 1
                wiggle_label = f"within_{wiggle_window_bp}bp"
                if skiver_ref_name is not None and skiver_ref_pos is not None:
                    detected_truth_reference_positions[(skiver_ref_name, skiver_ref_pos)] += 1
            else:
                wiggle_false += 1
                wiggle_label = f"outside_{wiggle_window_bp}bp"

            skiver_label = ERROR_TYPE_NAMES[skiver_error_type]
            _increment_nested(detected_by_skiver_truth, skiver_label, truth_label)
            _increment_nested(
                detected_by_skiver_same_sequence,
                skiver_label,
                same_sequence_label,
            )
            _increment_nested(detected_by_skiver_wiggle, skiver_label, wiggle_label)
            _increment_nested(detected_by_mate_truth, str(read_meta["mate"]), truth_label)
            _increment_nested(
                detected_by_forward_truth,
                row.get("is_forward", "NA"),
                truth_label,
            )
            try:
                phred = int(row.get("phred", "-1"))
            except ValueError:
                phred = -1
            _increment_nested(detected_by_phred_truth, _phred_bin_label(phred), truth_label)

    window_true = sum(1 for obs_id in detected_obs_ids if obs_id in obs_ids_with_truth)
    distinct_detected_truth = len(detected_truth_coordinates)
    distinct_detected_ref_positions = len(detected_truth_reference_positions)
    coordinate_truth_errors = int(truth_maps.get("coordinate_truth_errors", 0))
    coordinate_true_positives = detected_coordinate_keys & truth_coordinate_keys
    coordinate_false_positives = detected_coordinate_keys - truth_coordinate_keys
    coordinate_false_negatives = truth_coordinate_keys - detected_coordinate_keys
    coordinate_union = detected_coordinate_keys | truth_coordinate_keys
    coordinate_true_negatives = max(
        comparable_fastq_coordinates - len(coordinate_union),
        0,
    )
    coordinate_precision = (
        len(coordinate_true_positives) / len(detected_coordinate_keys)
        if detected_coordinate_keys
        else float("nan")
    )
    coordinate_recall = (
        len(coordinate_true_positives) / len(truth_coordinate_keys)
        if truth_coordinate_keys
        else float("nan")
    )
    coordinate_specificity = (
        coordinate_true_negatives
        / (coordinate_true_negatives + len(coordinate_false_positives))
        if coordinate_true_negatives + len(coordinate_false_positives)
        else float("nan")
    )
    covered_truth_coordinate_keys = truth_coordinate_keys & skiver_covered_coordinate_keys
    covered_true_positives = detected_coordinate_keys & covered_truth_coordinate_keys
    covered_false_positives = detected_coordinate_keys - covered_truth_coordinate_keys
    covered_false_negatives = covered_truth_coordinate_keys - detected_coordinate_keys
    covered_true_negatives = (
        skiver_covered_coordinate_keys
        - detected_coordinate_keys
        - covered_truth_coordinate_keys
    )
    covered_union = detected_coordinate_keys | covered_truth_coordinate_keys
    covered_precision = (
        len(covered_true_positives) / len(detected_coordinate_keys)
        if detected_coordinate_keys
        else float("nan")
    )
    covered_recall = (
        len(covered_true_positives) / len(covered_truth_coordinate_keys)
        if covered_truth_coordinate_keys
        else float("nan")
    )
    covered_specificity = (
        len(covered_true_negatives)
        / (len(covered_true_negatives) + len(covered_false_positives))
        if len(covered_true_negatives) + len(covered_false_positives)
        else float("nan")
    )
    covered_f1 = (
        2 * covered_precision * covered_recall / (covered_precision + covered_recall)
        if covered_precision + covered_recall > 0
        else float("nan")
    )
    nearest_distance_summary = {
        "n": int(len(nearest_truth_distances)),
        "min": int(min(nearest_truth_distances)) if nearest_truth_distances else None,
        "median": float(np.median(nearest_truth_distances)) if nearest_truth_distances else None,
        "p95": float(np.percentile(nearest_truth_distances, 95)) if nearest_truth_distances else None,
        "max": int(max(nearest_truth_distances)) if nearest_truth_distances else None,
    }
    nearest_distance_histogram = [
        {"max_distance_bp": upper, "count": int(count)}
        for upper, count in zip(
            [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
            np.histogram(
                nearest_truth_distances,
                bins=[-0.5, 0.5, 1.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
            )[0] if nearest_truth_distances else [0] * 10,
        )
    ]
    return {
        "split": split,
        "available": True,
        "base_observations": str(base_observations),
        "bam": str(bam_path),
        "fastqs": [str(path) for path in reads],
        "wiggle_window_bp": wiggle_window_bp,
        "total_base_rows": total_rows,
        "skiver_detected_errors": detected_errors,
        "same_coordinate_true": same_coordinate_true,
        "same_coordinate_false": same_coordinate_false,
        "same_sequence_true": same_sequence_true,
        "same_sequence_false": same_sequence_false,
        "same_sequence_true_fraction": (
            same_sequence_true / detected_errors if detected_errors else float("nan")
        ),
        "wiggle_window_true": wiggle_true,
        "wiggle_window_false": wiggle_false,
        "wiggle_window_true_fraction": (
            wiggle_true / detected_errors if detected_errors else float("nan")
        ),
        "unmapped_detected": unmapped_detected,
        "invalid_coordinate_rows": invalid_coordinate_rows,
        "strict_true_fraction": (
            same_coordinate_true / detected_errors if detected_errors else float("nan")
        ),
        "window_true": window_true,
        "window_false": detected_errors - window_true,
        "window_true_fraction": (
            window_true / detected_errors if detected_errors else float("nan")
        ),
        "coordinate_truth_errors": coordinate_truth_errors,
        "deletion_truth_errors": int(truth_maps.get("deletion_truth_errors", 0)),
        "distinct_bam_truth_errors_detected": distinct_detected_truth,
        "distinct_bam_reference_positions_detected_within_wiggle": (
            distinct_detected_ref_positions
        ),
        "coordinate_truth_error_detection_fraction": (
            distinct_detected_truth / coordinate_truth_errors
            if coordinate_truth_errors
            else float("nan")
        ),
        "duplication_factor": (
            same_coordinate_true / distinct_detected_truth
            if distinct_detected_truth
            else float("nan")
        ),
        "match_modes": {
            "strict_coordinate": {
                "true": same_coordinate_true,
                "false": detected_errors - same_coordinate_true,
                "true_fraction": (
                    same_coordinate_true / detected_errors
                    if detected_errors
                    else float("nan")
                ),
            },
            "same_sequence": {
                "true": same_sequence_true,
                "false": same_sequence_false,
                "true_fraction": (
                    same_sequence_true / detected_errors
                    if detected_errors
                    else float("nan")
                ),
            },
            f"wiggle_{wiggle_window_bp}bp": {
                "true": wiggle_true,
                "false": wiggle_false,
                "true_fraction": (
                    wiggle_true / detected_errors if detected_errors else float("nan")
                ),
            },
        },
        "nearest_truth_distance_summary": nearest_distance_summary,
        "nearest_truth_distance_histogram": nearest_distance_histogram,
        "coordinate_truth_by_type": truth_maps.get("coordinate_truth_by_type", {}),
        "coordinate_set_confusion": {
            "universe": int(comparable_fastq_coordinates),
            "skiver_positive_coordinates": int(len(detected_coordinate_keys)),
            "bam_truth_positive_coordinates": int(len(truth_coordinate_keys)),
            "true_positive": int(len(coordinate_true_positives)),
            "false_positive": int(len(coordinate_false_positives)),
            "false_negative": int(len(coordinate_false_negatives)),
            "true_negative": int(coordinate_true_negatives),
            "precision": coordinate_precision,
            "recall": coordinate_recall,
            "specificity": coordinate_specificity,
            "jaccard": (
                len(coordinate_true_positives) / len(coordinate_union)
                if coordinate_union
                else float("nan")
            ),
        },
        "skiver_covered_coordinate_confusion": {
            "universe": int(len(skiver_covered_coordinate_keys)),
            "skiver_positive_coordinates": int(len(detected_coordinate_keys)),
            "bam_truth_positive_coordinates": int(len(covered_truth_coordinate_keys)),
            "total_bam_truth_positive_coordinates": int(len(truth_coordinate_keys)),
            "out_of_scope_due_to_skiver_sparsity": int(
                len(truth_coordinate_keys - skiver_covered_coordinate_keys)
            ),
            "skiver_covered_coordinate_fraction": (
                len(skiver_covered_coordinate_keys) / comparable_fastq_coordinates
                if comparable_fastq_coordinates
                else float("nan")
            ),
            "bam_truth_coverage_fraction": (
                len(covered_truth_coordinate_keys) / len(truth_coordinate_keys)
                if truth_coordinate_keys
                else float("nan")
            ),
            "true_positive": int(len(covered_true_positives)),
            "false_positive": int(len(covered_false_positives)),
            "false_negative": int(len(covered_false_negatives)),
            "true_negative": int(len(covered_true_negatives)),
            "precision": covered_precision,
            "recall": covered_recall,
            "specificity": covered_specificity,
            "f1": covered_f1,
            "jaccard": (
                len(covered_true_positives) / len(covered_union)
                if covered_union
                else float("nan")
            ),
        },
        "duplicate_name_mate_keys": int(truth_maps.get("duplicate_name_mate_keys", 0)),
        "max_name_mate_occurrences": int(
            truth_maps.get("max_name_mate_occurrences", 0)
        ),
        "detected_by_skiver_type_and_truth": _nested_counter_rows(
            detected_by_skiver_truth,
            "skiver_error_type",
            "bam_truth_class",
        ),
        "detected_by_skiver_type_and_same_sequence_truth": _nested_counter_rows(
            detected_by_skiver_same_sequence,
            "skiver_error_type",
            "same_sequence_class",
        ),
        "detected_by_skiver_type_and_wiggle_truth": _nested_counter_rows(
            detected_by_skiver_wiggle,
            "skiver_error_type",
            "wiggle_class",
        ),
        "detected_by_mate_and_truth": _nested_counter_rows(
            detected_by_mate_truth,
            "mate",
            "bam_truth_class",
        ),
        "detected_by_skiver_orientation_and_truth": _nested_counter_rows(
            detected_by_forward_truth,
            "skiver_is_forward",
            "bam_truth_class",
        ),
        "detected_by_phred_bin_and_truth": _nested_counter_rows(
            detected_by_phred_truth,
            "phred_bin",
            "bam_truth_class",
        ),
        "top_truth_coordinates_by_skiver_detections": [
            {
                "read_id": int(read_id),
                "physical_pos": int(physical_pos),
                "skiver_detections": int(count),
            }
            for (read_id, physical_pos), count in detected_truth_coordinates.most_common(25)
        ],
    }


def _skiver_bam_error_match_metrics(
    *,
    reference: Path,
    train_prefix: Path,
    test_prefix: Path,
    train_reads: Sequence[Path],
    test_reads: Sequence[Path],
    train_bam: Path,
    test_bam: Path,
) -> dict[str, object]:
    """Return Skiver-vs-BAM truth diagnostic metrics."""
    return {
        "schema_version": 1,
        "description": (
            "Compares Skiver-detected base-observation errors to genome-blender "
            "BAM truth at the same physical FASTQ coordinate, on the same "
            "reference sequence, and within a configurable reference-coordinate "
            "wiggle window."
        ),
        "splits": _build_bam_error_match_splits(
            reference=reference,
            train_prefix=train_prefix,
            test_prefix=test_prefix,
            train_reads=train_reads,
            test_reads=test_reads,
            train_bam=train_bam,
            test_bam=test_bam,
        ),
    }


def _build_bam_error_match_splits(
    *,
    reference: Path,
    train_prefix: Path,
    test_prefix: Path,
    train_reads: Sequence[Path],
    test_reads: Sequence[Path],
    train_bam: Path,
    test_bam: Path,
) -> list[dict[str, object]]:
    """Build the train/test BAM error-match splits concurrently (independent)."""
    split_args = [
        ("train", train_prefix, train_reads, train_bam),
        ("test", test_prefix, test_reads, test_bam),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(
                _skiver_bam_error_match_split,
                split=split,
                base_observations=prefix.with_suffix(".base_observations.tsv"),
                reads=reads,
                bam_path=bam,
                reference=reference,
            )
            for split, prefix, reads, bam in split_args
        ]
        return [fut.result() for fut in futs]


def _count_summary(counts: np.ndarray) -> dict[str, object]:
    """Return a compact summary for one event-count vector."""
    values = np.asarray(counts, dtype=np.float64)
    return {
        "total": float(values.sum()),
        "counts": values.tolist(),
        "probs": _normalise(values).tolist(),
        "error_rate": _error_rate(values),
    }


def _split_rate_recovery(
    *,
    split: str,
    source_counts: np.ndarray,
    retrained_counts: np.ndarray,
    observed_counts: np.ndarray,
    physical_truth: dict[str, object],
) -> dict[str, object]:
    """Return source/retrained model marginal recovery metrics for one split."""
    source_rate = _error_rate(source_counts)
    retrained_rate = _error_rate(retrained_counts)
    rate_delta = retrained_rate - source_rate
    return {
        "split": split,
        "source_model": _count_summary(source_counts),
        "retrained_model": _count_summary(retrained_counts),
        "skiver_observed_window": _count_summary(observed_counts),
        "physical_bam_truth": physical_truth,
        "source_model_error_rate": source_rate,
        "retrained_model_error_rate": retrained_rate,
        "model_error_rate_delta": rate_delta,
        "model_error_rate_ratio": (
            retrained_rate / source_rate if source_rate > 0.0 else float("nan")
        ),
        "model_retrained_vs_source_tv": _tv(retrained_counts, source_counts),
        "model_retrained_vs_source_kl": _kl(retrained_counts, source_counts),
        "skiver_observed_vs_source_tv": _tv(observed_counts, source_counts),
        "skiver_observed_vs_source_kl": _kl(observed_counts, source_counts),
        "skiver_observed_vs_retrained_tv": _tv(observed_counts, retrained_counts),
        "skiver_observed_vs_retrained_kl": _kl(observed_counts, retrained_counts),
    }


def _rate_recovery_metrics(
    *,
    source_model: Path,
    retrained_model: Path,
    reference: Path,
    use_vi_source: bool,
    train_records: Sequence[tuple[int, int, int, int]],
    test_records: Sequence[tuple[int, int, int, int]],
    train_bam: Path,
    test_bam: Path,
    v: int = 13,
    observed_weibull: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return train/test rate-recovery metrics."""
    split_records = {"train": train_records, "test": test_records}
    split_bams = {"train": train_bam, "test": test_bam}
    splits = []
    for split, records in split_records.items():
        source_counts = _expected_counts(source_model, records, use_vi=use_vi_source)
        retrained_counts = _expected_counts(retrained_model, records)
        observed_counts = _observed_counts(records)
        splits.append(
            _split_rate_recovery(
                split=split,
                source_counts=source_counts,
                retrained_counts=retrained_counts,
                observed_counts=observed_counts,
                physical_truth=_physical_truth_from_bam(split_bams[split], reference),
            )
        )

    return {
        "schema_version": 1,
        "description": (
            "Model marginal rates compare source and retrained models on the "
            "same Skiver context/base exposure. Skiver observed window rates "
            "are overlapping dump observations. Physical BAM truth rates are "
            "per-read-base generator truth from genome-blender BAMs."
        ),
        "error_type_names": list(ERROR_TYPE_NAMES),
        "source_model": str(source_model),
        "retrained_model": str(retrained_model),
        "reference": str(reference),
        "source_uses_vi": use_vi_source,
        "v": v,
        "observed_weibull": observed_weibull,
        "splits": splits,
    }


def _fit_phred_calibration(
    base_observations: Sequence[Path],
    *,
    include_outliers: bool = True,
) -> dict[str, object]:
    """Fit P(Q | error_type) calibration from Skiver base-observation TSVs."""
    counts = np.zeros((NUM_ERROR_TYPES, N_PHRED), dtype=np.int64)
    n_observations = 0
    for path in base_observations:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if not include_outliers and row.get("passes_filter") == "false":
                    continue
                try:
                    phred = int(row.get("phred", "-1"))
                except ValueError:
                    continue
                if phred < 0:
                    continue
                error_type = encode_error_type(
                    row.get("true_base", "N"),
                    row.get("obs_base", "N"),
                    row.get("edit_op", "NA"),
                )
                q = min(max(phred, PHRED_MIN), PHRED_MAX)
                counts[error_type, q] += 1
                n_observations += 1

    probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + N_PHRED)
    return {
        "platform": "synthetic",
        "split": "generated",
        "include_outliers": include_outliers,
        "error_type_names": list(ERROR_TYPE_NAMES),
        "phred_min": PHRED_MIN,
        "phred_max": PHRED_MAX,
        "counts": counts.tolist(),
        "probs": probs.astype(float).tolist(),
        "n_observations": int(n_observations),
    }


def _truth_error_type_index(truth_classes: Sequence[str], truth_events: Sequence[str]) -> int:
    """Return the encoded BAM-truth error type for one FASTQ coordinate."""
    if not any(truth_class in {"substitution", "insertion"} for truth_class in truth_classes):
        return 0
    for truth_event in truth_events:
        if ">" in truth_event:
            obs_base = truth_event.split(">", 1)[1]
            return 1 + BASE_TO_IDX.get(obs_base, 0)
        if truth_event.startswith("ins_") and len(truth_event) >= 5:
            return 5 + BASE_TO_IDX.get(truth_event[-1], 0)
        if truth_event == "insertion":
            return 5
    return 0


def _quality_artifact_from_records(
    records: Sequence[tuple[int, int]],
    *,
    split: str,
) -> dict[str, object]:
    """Return a P(Q | BAM truth error_type) artifact from covered records."""
    counts = np.zeros((NUM_ERROR_TYPES, N_PHRED), dtype=np.int64)
    for error_type, phred in records:
        q = min(max(phred, PHRED_MIN), PHRED_MAX)
        counts[error_type, q] += 1
    probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + N_PHRED)
    return {
        "platform": "synthetic",
        "split": split,
        "label_source": "bam_truth_on_skiver_covered_coordinates",
        "include_outliers": True,
        "error_type_names": list(ERROR_TYPE_NAMES),
        "phred_min": PHRED_MIN,
        "phred_max": PHRED_MAX,
        "counts": counts.tolist(),
        "probs": probs.astype(float).tolist(),
        "n_observations": int(len(records)),
    }


def _quality_calibration_artifact_from_records(
    records: Sequence[tuple[int, int]],
    *,
    models: Sequence[str],
    validation_fraction: float,
    seed: int,
    fit_bounds: bool,
    steps: int,
    lr: float,
) -> dict[str, object]:
    """Fit P(error | Q) from BAM-truth labels on Skiver-covered coordinates."""
    rng = random.Random(seed)
    train_total = np.zeros(N_PHRED, dtype=np.int64)
    train_error = np.zeros(N_PHRED, dtype=np.int64)
    validation_total = np.zeros(N_PHRED, dtype=np.int64)
    validation_error = np.zeros(N_PHRED, dtype=np.int64)
    for error_type, phred in records:
        q = min(max(phred, PHRED_MIN), PHRED_MAX)
        is_error = int(error_type != 0)
        if rng.random() < validation_fraction:
            validation_total[q] += 1
            validation_error[q] += is_error
        else:
            train_total[q] += 1
            train_error[q] += is_error
    counts = CalibrationCounts(
        train_total=train_total,
        train_error=train_error,
        validation_total=validation_total,
        validation_error=validation_error,
        raw_rows=len(records),
        unique_rows=len(records),
        duplicate_rows=0,
        conflicting_duplicates=0,
        skipped_rows=0,
    )
    fits = [
        _fit_model(
            counts,
            model=model,
            fit_bounds=fit_bounds,
            steps=steps,
            lr=lr,
        )
        for model in models
    ]
    selected = min(fits, key=lambda fit: fit.validation_aic)
    return {
        "artifact_type": "skiver_genome_blender_quality_calibration",
        "label_source": "bam_truth_on_skiver_covered_coordinates",
        "selected_model": selected.model,
        "selected_params": selected.params,
        "genome_blender_config": {
            "quality_calibration_model": selected.model,
            **selected.params,
        },
        "phred_min": PHRED_MIN,
        "phred_max": PHRED_MAX,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "passes_filter_only": False,
        "fit_bounds": fit_bounds,
        "input_paths": [],
        "deduplication": {
            "raw_rows": counts.raw_rows,
            "unique_rows": counts.unique_rows,
            "duplicate_rows": counts.duplicate_rows,
            "conflicting_duplicates": counts.conflicting_duplicates,
            "skipped_rows": counts.skipped_rows,
        },
        "counts_by_q": _counts_table(counts),
        "candidate_fits": _fit_summaries_to_json(fits),
    }


def _covered_truth_quality_records(
    *,
    base_observations: Path,
    reads: Sequence[Path],
    bam_path: Path,
    reference: Path,
) -> list[tuple[int, int]]:
    """Return deduplicated (BAM truth error_type, Q) records covered by Skiver."""
    truth_maps = _truth_maps_from_bam(bam_path, reference)
    if not truth_maps.get("available"):
        return []
    read_index = _fastq_read_index(reads)
    records: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    with open(base_observations, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("passes_filter") == "false":
                continue
            try:
                read_id = int(row["read_id"])
                read_pos = int(row["read_pos"])
                t = int(row["t"])
                phred = int(row.get("phred", "-1"))
            except (KeyError, ValueError):
                continue
            if phred < 0 or read_id < 0 or read_id >= len(read_index):
                continue
            read_meta = read_index[read_id]
            read_truth = _read_truth_for_meta(truth_maps, read_meta)
            if read_truth is None:
                continue
            physical_pos = _skiver_row_physical_pos(
                is_forward=row.get("is_forward") == "true",
                read_pos=read_pos,
                t=t,
            )
            if not 0 <= physical_pos < int(read_meta["read_length"]):
                continue
            key = (read_id, physical_pos)
            if key in seen:
                continue
            seen.add(key)
            truth_classes, truth_events = _truth_classes_at_pos(
                truth_maps,
                read_meta,
                physical_pos,
            )
            records.append((_truth_error_type_index(truth_classes, truth_events), phred))
    return records


def _covered_truth_quality_artifacts(
    *,
    split: str,
    base_observations: Path,
    reads: Sequence[Path],
    bam_path: Path,
    reference: Path,
    models: Sequence[str],
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return P(Q | true error_type) and P(error | Q) artifacts for covered bases."""
    records = _covered_truth_quality_records(
        base_observations=base_observations,
        reads=reads,
        bam_path=bam_path,
        reference=reference,
    )
    return (
        _quality_artifact_from_records(records, split=split),
        _quality_calibration_artifact_from_records(
            records,
            models=models,
            validation_fraction=0.2,
            seed=seed,
            fit_bounds=True,
            steps=2000,
            lr=0.03,
        ),
    )


def _mean_phred(calibration: dict[str, object]) -> np.ndarray:
    """Return mean Q for each error type in a calibration dict."""
    probs = np.asarray(calibration["probs"], dtype=np.float64)
    q_values = np.arange(
        int(calibration["phred_min"]),
        int(calibration["phred_max"]) + 1,
        dtype=np.float64,
    )
    return probs @ q_values


def _phred_calibration_metrics(
    source_calibration: dict[str, object] | None,
    train_calibration: dict[str, object],
    test_calibration: dict[str, object],
) -> dict[str, object]:
    """Compare source and recovered Phred calibration distributions."""
    result: dict[str, object] = {
        "train_n_observations": train_calibration["n_observations"],
        "test_n_observations": test_calibration["n_observations"],
        "error_type_names": list(ERROR_TYPE_NAMES),
        "train_mean_phred": _mean_phred(train_calibration).tolist(),
        "test_mean_phred": _mean_phred(test_calibration).tolist(),
        "q_values": list(range(PHRED_MIN, PHRED_MAX + 1)),
    }
    if source_calibration is None:
        result["source_available"] = False
        return result

    source_probs = np.asarray(source_calibration["probs"], dtype=np.float64)
    train_probs = np.asarray(train_calibration["probs"], dtype=np.float64)
    test_probs = np.asarray(test_calibration["probs"], dtype=np.float64)
    source_counts = np.asarray(source_calibration["counts"], dtype=np.float64)
    train_counts = np.asarray(train_calibration["counts"], dtype=np.float64)
    test_counts = np.asarray(test_calibration["counts"], dtype=np.float64)

    def marginal(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        if total <= 0.0:
            return np.zeros(counts.shape[1], dtype=np.float64)
        return counts.sum(axis=0) / total

    source_marginal = marginal(source_counts)
    train_marginal = marginal(train_counts)
    test_marginal = marginal(test_counts)
    result.update(
        {
            "source_available": True,
            "source_mean_phred": _mean_phred(source_calibration).tolist(),
            "source_marginal_q": source_marginal.tolist(),
            "train_marginal_q": train_marginal.tolist(),
            "test_marginal_q": test_marginal.tolist(),
            "train_vs_source_marginal_q_tv": float(
                0.5 * np.abs(train_marginal - source_marginal).sum()
            ),
            "test_vs_source_marginal_q_tv": float(
                0.5 * np.abs(test_marginal - source_marginal).sum()
            ),
            "test_vs_train_marginal_q_tv": float(
                0.5 * np.abs(test_marginal - train_marginal).sum()
            ),
            "train_vs_source_tv_by_error_type": (
                0.5 * np.abs(train_probs - source_probs).sum(axis=1)
            ).tolist(),
            "test_vs_source_tv_by_error_type": (
                0.5 * np.abs(test_probs - source_probs).sum(axis=1)
            ).tolist(),
            "test_vs_train_tv_by_error_type": (
                0.5 * np.abs(test_probs - train_probs).sum(axis=1)
            ).tolist(),
        }
    )
    return result


def _qcal_candidate_by_model(artifact: dict[str, object], model: str) -> dict[str, object]:
    """Return one candidate fit from a quality calibration artifact."""
    for fit in artifact.get("candidate_fits", []):
        if fit["model"] == model:
            return fit
    raise ValueError(f"Calibration artifact does not contain model {model!r}")


def _quality_curve_metrics(
    source_artifact: dict[str, object] | None,
    train_artifact: dict[str, object],
    test_artifact: dict[str, object],
) -> dict[str, object]:
    """Compare genome-blender Q-to-error calibration model artifacts."""
    selected_model = str(train_artifact["selected_model"])
    train_fit = _qcal_candidate_by_model(train_artifact, selected_model)
    test_fit = _qcal_candidate_by_model(test_artifact, selected_model)
    train_counts_by_q = train_artifact["counts_by_q"]
    test_counts_by_q = test_artifact["counts_by_q"]

    def weights_for(rows: Sequence[dict[str, object]]) -> np.ndarray:
        weights = np.asarray([row["n"] for row in rows], dtype=np.float64)
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights

    train_weights = weights_for(train_counts_by_q)
    test_weights = weights_for(test_counts_by_q)
    train_curve = np.asarray(train_fit["fitted_error_rate_by_q"], dtype=np.float64)
    test_curve = np.asarray(test_fit["fitted_error_rate_by_q"], dtype=np.float64)
    result: dict[str, object] = {
        "selected_model": selected_model,
        "train_params": train_fit["params"],
        "test_params": test_fit["params"],
        "occupied_q_values": [
            int(row["q"]) for row in train_counts_by_q if int(row.get("n", 0)) > 0
        ],
        "unoccupied_q_values": [
            int(row["q"]) for row in train_counts_by_q if int(row.get("n", 0)) == 0
        ],
        "test_vs_train_weighted_abs_diff": float(
            (test_weights * np.abs(test_curve - train_curve)).sum()
        ),
        "train_validation_nll": train_fit["validation_nll"],
        "test_validation_nll": test_fit["validation_nll"],
    }
    if source_artifact is not None:
        source_model = str(source_artifact["selected_model"])
        source_fit = _qcal_candidate_by_model(source_artifact, source_model)
        source_curve = np.asarray(source_fit["fitted_error_rate_by_q"], dtype=np.float64)
        result.update(
            {
                "source_model": source_model,
                "source_params": source_fit["params"],
                "test_vs_source_weighted_abs_diff": float(
                    (test_weights * np.abs(test_curve - source_curve)).sum()
                ),
                "train_vs_source_weighted_abs_diff": float(
                    (train_weights * np.abs(train_curve - source_curve)).sum()
                ),
            }
        )
        if source_model == selected_model:
            source_params = source_fit["params"]
            result["parameter_deltas_train_minus_source"] = {
                key: float(train_fit["params"][key] - source_params[key])
                for key in sorted(source_params)
                if key in train_fit["params"]
            }
            result["parameter_deltas_test_minus_source"] = {
                key: float(test_fit["params"][key] - source_params[key])
                for key in sorted(source_params)
                if key in test_fit["params"]
            }
    return result


def _expected_counts(
    model_path: Path,
    records: Sequence[tuple[int, int, int, int]],
    *,
    use_vi: bool = False,
) -> np.ndarray:
    """Return model-expected event counts under observed context/base exposure."""
    model = load_model(model_path, use_vi=use_vi)
    opportunities: dict[tuple[int, int], int] = defaultdict(int)
    for context_index, true_base, _, count in records:
        opportunities[(context_index, true_base)] += count

    expected = np.zeros(NUM_ERROR_TYPES, dtype=np.float64)
    for (context_index, true_base), count in opportunities.items():
        expected += count * probabilities_for_context(model, context_index, true_base)
    return expected


def _normalise(counts: np.ndarray) -> np.ndarray:
    """Return counts normalised to probabilities."""
    total = counts.sum()
    if total <= 0.0:
        raise ValueError("Cannot normalise empty counts")
    return counts / total


def _kl(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Return KL divergence KL(p || q) after normalising count vectors."""
    eps = 1e-12
    p = np.clip(_normalise(p_counts), eps, 1.0)
    q = np.clip(_normalise(q_counts), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _tv(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Return total variation distance after normalising count vectors."""
    return float(0.5 * np.abs(_normalise(p_counts) - _normalise(q_counts)).sum())


def _error_rate(counts: np.ndarray) -> float:
    """Return non-match event fraction."""
    probs = _normalise(counts)
    return float(1.0 - probs[0])


def _metrics(
    *,
    source_counts: np.ndarray,
    observed_counts: np.ndarray,
    retrained_counts: np.ndarray,
) -> dict[str, object]:
    """Return recovery metrics for source, observed, and retrained model."""
    return {
        "source_error_rate": _error_rate(source_counts),
        "observed_error_rate": _error_rate(observed_counts),
        "retrained_error_rate": _error_rate(retrained_counts),
        "observed_vs_source_tv": _tv(observed_counts, source_counts),
        "observed_vs_source_kl": _kl(observed_counts, source_counts),
        "retrained_vs_source_tv": _tv(retrained_counts, source_counts),
        "retrained_vs_source_kl": _kl(retrained_counts, source_counts),
        "observed_vs_retrained_tv": _tv(observed_counts, retrained_counts),
        "observed_vs_retrained_kl": _kl(observed_counts, retrained_counts),
        "source_probs": _normalise(source_counts).tolist(),
        "observed_probs": _normalise(observed_counts).tolist(),
        "retrained_probs": _normalise(retrained_counts).tolist(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark recovery of a context error model on synthetic reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Source model .pt.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference FASTA/FASTQ.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--skiver-bin",
        type=Path,
        default=Path("target/debug/skiver"),
    )
    parser.add_argument(
        "--genome-blender-dir",
        type=Path,
        default=Path("../genome-blender"),
        help="Path to the genome-blender checkout.",
    )
    parser.add_argument(
        "--genome-blender-conda-env",
        default="genome_blender_dev",
        help=(
            "Conda environment used to run genome-blender. Set to an empty "
            "string to use the current Python."
        ),
    )
    parser.add_argument(
        "--genome-blender-python",
        type=Path,
        default=None,
        help="Explicit Python executable for genome-blender; overrides conda env.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip genome-blender generation and use --train-reads/--test-reads.",
    )
    parser.add_argument(
        "--train-reads",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Pre-generated training FASTQ(s) used with --skip-simulation. "
            "Pass both R1 and R2 files for paired-end data."
        ),
    )
    parser.add_argument(
        "--test-reads",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Pre-generated test FASTQ(s) used with --skip-simulation. "
            "Pass both R1 and R2 files for paired-end data."
        ),
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=100,
        help=(
            "Number of genome-blender amplicon reads generated "
            "for each train/test split."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--v", type=int, default=13)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--use-vi",
        action="store_true",
        help="Generate from VI source params.",
    )
    parser.add_argument(
        "--phred-calibration",
        type=Path,
        default=None,
        help=(
            "Deprecated alias for --joint-phred-calibration-json. "
            "Empirical P(Q | error_type) model used for generation."
        ),
    )
    parser.add_argument(
        "--joint-phred-calibration-json",
        type=Path,
        default=None,
        help=(
            "Empirical joint-quality model, represented as P(Q | error_type). "
            "When provided, genome-blender samples Q after sampling the Skiver "
            "error type."
        ),
    )
    parser.add_argument(
        "--quality-calibration-model-json",
        type=Path,
        default=None,
        help=(
            "Skiver-trained genome-blender Q-to-error calibration artifact. "
            "Used as a validation target only when --joint-phred-calibration-json "
            "is provided."
        ),
    )
    parser.add_argument(
        "--quality-calibration-fit-model",
        choices=("log-linear", "sigmoid", "both"),
        default="log-linear",
        help="Candidate model(s) to refit from generated Skiver dumps.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--pseudo-count", type=float, default=0.5)
    parser.add_argument(
        "--vi-steps",
        type=int,
        default=500,
        help="Number of VI retraining steps; set 0 to skip VI uncertainty output.",
    )
    parser.add_argument("--vi-lr", type=float, default=0.01)
    parser.add_argument("--vi-prior-scale", type=float, default=2.0)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument(
        "--weibull-outlier-threshold",
        type=float,
        # 1e-3 is approximately Bonferroni-corrected (p<0.05 / ~50-100 keys).
        # The default skiver value of 1e-9 is too conservative for amplicon
        # datasets and fails to remove read-end-clustered keys that inflate λ.
        default=1e-3,
        help=(
            "P-value threshold for skiver's iterative Binomial outlier filter "
            "when estimating the Weibull calibration rate. Keys with anomalously "
            "high error counts (P(X<=obs) < threshold under fitted Weibull) are "
            "excluded before the final λ/β fit. Default 1e-3 is ~Bonferroni-"
            "corrected at p<0.05 for typical amplicon key counts."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic generate-and-recover benchmark."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    args.model = args.model.resolve()
    args.reference = args.reference.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.skiver_bin.is_absolute():
        args.skiver_bin = REPO_ROOT / args.skiver_bin
    args.skiver_bin = args.skiver_bin.resolve()
    if not args.genome_blender_dir.is_absolute():
        args.genome_blender_dir = (REPO_ROOT / args.genome_blender_dir).resolve()
    else:
        args.genome_blender_dir = args.genome_blender_dir.resolve()
    if args.genome_blender_python is not None:
        args.genome_blender_python = args.genome_blender_python.resolve()
    if args.genome_blender_conda_env == "":
        args.genome_blender_conda_env = None
    if args.train_reads is not None:
        args.train_reads = [path.resolve() for path in args.train_reads]
    if args.test_reads is not None:
        args.test_reads = [path.resolve() for path in args.test_reads]
    if args.phred_calibration is not None:
        args.phred_calibration = args.phred_calibration.resolve()
    if args.joint_phred_calibration_json is not None:
        args.joint_phred_calibration_json = (
            args.joint_phred_calibration_json.resolve()
        )
    if (
        args.phred_calibration is not None
        and args.joint_phred_calibration_json is not None
        and args.phred_calibration != args.joint_phred_calibration_json
    ):
        raise ValueError(
            "--phred-calibration and --joint-phred-calibration-json point to "
            "different files; provide only the joint-quality model path."
        )
    joint_phred_calibration = (
        args.joint_phred_calibration_json
        if args.joint_phred_calibration_json is not None
        else args.phred_calibration
    )
    if args.quality_calibration_model_json is not None:
        args.quality_calibration_model_json = (
            args.quality_calibration_model_json.resolve()
        )
    quality_calibration_config = _load_quality_calibration_model(
        args.quality_calibration_model_json
    )

    model_id, parameterization, context_length = _load_artifact_metadata(args.model)
    additive_context = parameterization == "additive_context"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_prefix_sim = args.output_dir / "synthetic_train"
    test_prefix_sim = args.output_dir / "synthetic_test"
    train_bam = train_prefix_sim.with_suffix(".bam")
    test_bam = test_prefix_sim.with_suffix(".bam")
    train_input_csv = args.output_dir / "synthetic_train.genomes.csv"
    test_input_csv = args.output_dir / "synthetic_test.genomes.csv"
    train_prefix = args.output_dir / "dump_train"
    test_prefix = args.output_dir / "dump_test"
    retrained_model = args.output_dir / "retrained_context_model.pt"
    metrics_path = args.output_dir / "recovery_metrics.json"
    rate_metrics_path = args.output_dir / "rate_recovery_metrics.json"
    skiver_bam_match_path = args.output_dir / "skiver_bam_error_match_metrics.json"
    vi_metrics_path = args.output_dir / "vi_uncertainty_metrics.json"
    source_phred_path = args.output_dir / "source_phred_calibration.json"
    recovered_train_phred_path = args.output_dir / "recovered_train_phred_calibration.json"
    recovered_test_phred_path = args.output_dir / "recovered_test_phred_calibration.json"
    covered_train_phred_path = args.output_dir / "covered_train_truth_phred_calibration.json"
    covered_test_phred_path = args.output_dir / "covered_test_truth_phred_calibration.json"
    phred_metrics_path = args.output_dir / "phred_calibration_metrics.json"
    source_qcal_path = args.output_dir / "source_quality_calibration_model.json"
    recovered_train_qcal_path = (
        args.output_dir / "recovered_train_quality_calibration_model.json"
    )
    recovered_test_qcal_path = (
        args.output_dir / "recovered_test_quality_calibration_model.json"
    )
    covered_train_qcal_path = (
        args.output_dir / "covered_train_truth_quality_calibration_model.json"
    )
    covered_test_qcal_path = (
        args.output_dir / "covered_test_truth_quality_calibration_model.json"
    )
    qcal_metrics_path = args.output_dir / "quality_calibration_model_metrics.json"

    if args.skip_simulation:
        if args.train_reads is None or args.test_reads is None:
            raise ValueError("--skip-simulation requires --train-reads and --test-reads")
        train_reads = args.train_reads
        test_reads = args.test_reads
    else:
        genome_blender_cmd = _genome_blender_command(
            genome_blender_dir=args.genome_blender_dir,
            conda_env=args.genome_blender_conda_env,
            python_executable=args.genome_blender_python,
        )
        train_reads = [
            _simulate_with_genome_blender(
                model=args.model,
                reference=args.reference,
                output_prefix=train_prefix_sim,
                input_csv=train_input_csv,
                num_reads=args.n_copies,
                seed=args.seed,
                joint_phred_calibration=joint_phred_calibration,
                quality_calibration_model=(
                    None
                    if joint_phred_calibration is not None
                    else quality_calibration_config
                ),
                use_vi=args.use_vi,
                genome_blender_cmd=genome_blender_cmd,
            )
        ]
        test_reads = [
            _simulate_with_genome_blender(
                model=args.model,
                reference=args.reference,
                output_prefix=test_prefix_sim,
                input_csv=test_input_csv,
                num_reads=args.n_copies,
                seed=args.seed + 1,
                joint_phred_calibration=joint_phred_calibration,
                quality_calibration_model=(
                    None
                    if joint_phred_calibration is not None
                    else quality_calibration_config
                ),
                use_vi=args.use_vi,
                genome_blender_cmd=genome_blender_cmd,
            )
        ]

    skiver_cmd = _skiver_command(args.skiver_bin)
    # The two `skiver dump`s and the `skiver analyze` are independent (distinct
    # output prefixes) and each spends most of its time in a child process, so run
    # them concurrently. Threads suffice — the GIL is released while the
    # subprocess runs. (Profiling showed these sequential subprocess waits were
    # the dominant Python wall-time in the benchmark.)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        dump_futs = [
            pool.submit(
                _dump,
                skiver_cmd=skiver_cmd,
                reads=reads,
                prefix=prefix,
                k=args.k,
                v=args.v,
                c=args.c,
                forward_only=args.forward_only,
            )
            for reads, prefix in ((train_reads, train_prefix), (test_reads, test_prefix))
        ]
        weibull_fut = pool.submit(
            _analyze_weibull_rate,
            skiver_cmd=skiver_cmd,
            reads=train_reads,
            prefix=train_prefix.parent / f"{train_prefix.name}_weibull",
            k=args.k,
            v=args.v,
            c=args.c,
            forward_only=args.forward_only,
            weibull_outlier_threshold=args.weibull_outlier_threshold,
        )
        for fut in dump_futs:
            fut.result()
        weibull_result = weibull_fut.result()
    weibull_rate: float | None = weibull_result["rate"] if weibull_result else None  # type: ignore[assignment]

    train_counts, test_counts = _counts_for_training(
        train_prefix=train_prefix,
        test_prefix=test_prefix,
        context_length=context_length,
        additive_context=additive_context,
        max_contexts=args.max_contexts,
    )

    source_artifact = _load_artifact(args.model)
    source_params = source_artifact["maximum_likelihood"]["params"]  # type: ignore[index]
    source_rate = compute_marginal_error_rate(
        train_counts.counts, source_params,  # type: ignore[arg-type]
        run_values=train_counts.run_values,
        additive_context=additive_context,
        context_indices=train_counts.context_indices,
    )
    logger.info("Source model marginal error rate on training data: %.6f", source_rate)
    if weibull_rate is not None:
        logger.info("Weibull window-averaged rate: %.6f (not used as calibration target)", weibull_rate)

    fit = fit_and_test(
        train_counts,
        test_counts,
        lr=args.lr,
        num_steps=args.steps,
        clip_norm=args.clip_norm,
        pseudo_count=args.pseudo_count,
        seed=args.seed,
    )
    vi_fit = None
    if args.vi_steps > 0:
        vi_fit = fit_bayesian_and_test(
            train_counts,
            test_counts,
            lr=args.vi_lr,
            num_steps=args.vi_steps,
            clip_norm=args.clip_norm,
            pseudo_count=args.pseudo_count,
            prior_scale=args.vi_prior_scale,
            seed=args.seed,
        )
    _save_retrained_artifact(
        retrained_model,
        model_id=model_id,
        parameterization=parameterization,
        context_length=context_length,
        fit_params=fit.params,
        fit_losses=fit.losses,
        train_log_likelihood=fit.train_log_likelihood,
        test_log_likelihood=fit.test_log_likelihood,
        vi_fit=vi_fit,
        train_counts=train_counts,
        test_counts=test_counts,
        v=args.v,
        weibull_rate=weibull_rate,
    )

    train_phred = _fit_phred_calibration(
        [train_prefix.with_suffix(".base_observations.tsv")],
        include_outliers=True,
    )
    test_phred = _fit_phred_calibration(
        [test_prefix.with_suffix(".base_observations.tsv")],
        include_outliers=True,
    )
    with open(recovered_train_phred_path, "w") as handle:
        json.dump(train_phred, handle, indent=2)
        handle.write("\n")
    with open(recovered_test_phred_path, "w") as handle:
        json.dump(test_phred, handle, indent=2)
        handle.write("\n")

    source_phred = None
    if joint_phred_calibration is not None:
        shutil.copyfile(joint_phred_calibration, source_phred_path)
        with open(joint_phred_calibration) as handle:
            source_phred = json.load(handle)
    phred_metrics = _phred_calibration_metrics(source_phred, train_phred, test_phred)
    with open(phred_metrics_path, "w") as handle:
        json.dump(phred_metrics, handle, indent=2)
        handle.write("\n")

    qcal_models = (
        ("log-linear", "sigmoid")
        if args.quality_calibration_fit_model == "both"
        else (args.quality_calibration_fit_model,)
    )
    covered_train_phred, covered_train_qcal = _covered_truth_quality_artifacts(
        split="train_bam_truth_skiver_covered",
        base_observations=train_prefix.with_suffix(".base_observations.tsv"),
        reads=train_reads,
        bam_path=train_bam,
        reference=args.reference,
        models=qcal_models,
        seed=args.seed,
    )
    covered_test_phred, covered_test_qcal = _covered_truth_quality_artifacts(
        split="test_bam_truth_skiver_covered",
        base_observations=test_prefix.with_suffix(".base_observations.tsv"),
        reads=test_reads,
        bam_path=test_bam,
        reference=args.reference,
        models=qcal_models,
        seed=args.seed + 1,
    )
    with open(covered_train_phred_path, "w") as handle:
        json.dump(covered_train_phred, handle, indent=2)
        handle.write("\n")
    with open(covered_test_phred_path, "w") as handle:
        json.dump(covered_test_phred, handle, indent=2)
        handle.write("\n")
    with open(covered_train_qcal_path, "w") as handle:
        json.dump(covered_train_qcal, handle, indent=2)
        handle.write("\n")
    with open(covered_test_qcal_path, "w") as handle:
        json.dump(covered_test_qcal, handle, indent=2)
        handle.write("\n")
    train_qcal = fit_quality_calibration(
        [train_prefix.with_suffix(".base_observations.tsv")],
        models=qcal_models,
        validation_fraction=0.2,
        seed=args.seed,
        passes_filter_only=False,
        max_rows=None,
        fit_bounds=True,
        steps=2000,
        lr=0.03,
    )
    test_qcal = fit_quality_calibration(
        [test_prefix.with_suffix(".base_observations.tsv")],
        models=qcal_models,
        validation_fraction=0.2,
        seed=args.seed + 1,
        passes_filter_only=False,
        max_rows=None,
        fit_bounds=True,
        steps=2000,
        lr=0.03,
    )
    source_qcal = None
    if args.quality_calibration_model_json is not None:
        shutil.copyfile(args.quality_calibration_model_json, source_qcal_path)
        with open(args.quality_calibration_model_json) as handle:
            source_qcal = json.load(handle)
    with open(recovered_train_qcal_path, "w") as handle:
        json.dump(train_qcal, handle, indent=2)
        handle.write("\n")
    with open(recovered_test_qcal_path, "w") as handle:
        json.dump(test_qcal, handle, indent=2)
        handle.write("\n")
    qcal_metrics = _quality_curve_metrics(source_qcal, train_qcal, test_qcal)
    with open(qcal_metrics_path, "w") as handle:
        json.dump(qcal_metrics, handle, indent=2)
        handle.write("\n")

    vi_metrics = _vi_uncertainty_metrics(args.model, retrained_model)
    with open(vi_metrics_path, "w") as handle:
        json.dump(vi_metrics, handle, indent=2)
        handle.write("\n")

    train_records = _observed_records_from_dump(
        train_prefix.with_suffix(".base_observations.tsv"),
        context_length=context_length,
    )
    test_records = _observed_records_from_dump(
        test_prefix.with_suffix(".base_observations.tsv"),
        context_length=context_length,
    )
    rate_metrics = _rate_recovery_metrics(
        source_model=args.model,
        retrained_model=retrained_model,
        reference=args.reference,
        use_vi_source=args.use_vi,
        train_records=train_records,
        test_records=test_records,
        train_bam=train_bam,
        test_bam=test_bam,
        v=args.v,
        observed_weibull=weibull_result,
    )
    with open(rate_metrics_path, "w") as handle:
        json.dump(rate_metrics, handle, indent=2)
        handle.write("\n")

    skiver_bam_match = _skiver_bam_error_match_metrics(
        reference=args.reference,
        train_prefix=train_prefix,
        test_prefix=test_prefix,
        train_reads=train_reads,
        test_reads=test_reads,
        train_bam=train_bam,
        test_bam=test_bam,
    )
    with open(skiver_bam_match_path, "w") as handle:
        json.dump(skiver_bam_match, handle, indent=2)
        handle.write("\n")

    source_expected = _expected_counts(args.model, test_records, use_vi=args.use_vi)
    observed = _observed_counts(test_records)
    retrained_expected = _expected_counts(retrained_model, test_records)
    metrics = _metrics(
        source_counts=source_expected,
        observed_counts=observed,
        retrained_counts=retrained_expected,
    )
    metrics.update(
        {
            "simulator": "genome-blender",
            "simulation_skipped": args.skip_simulation,
            "genome_blender_dir": str(args.genome_blender_dir),
            "genome_blender_conda_env": args.genome_blender_conda_env,
            "source_model": str(args.model),
            "retrained_model": str(retrained_model),
            "parameterization": parameterization,
            "context_length": context_length,
            "n_reads_per_split": args.n_copies,
            "n_train_observations": train_counts.total_observations,
            "n_test_observations": test_counts.total_observations,
            "train_log_likelihood": fit.train_log_likelihood,
            "test_log_likelihood": fit.test_log_likelihood,
            "vi_enabled": vi_fit is not None,
            "rate_recovery_metrics": str(rate_metrics_path),
            "skiver_bam_error_match_metrics": str(skiver_bam_match_path),
            "vi_uncertainty_metrics": str(vi_metrics_path),
            "source_phred_calibration": (
                str(source_phred_path) if joint_phred_calibration is not None else None
            ),
            "generation_quality_model": (
                "empirical_joint_p_q_given_error_type"
                if joint_phred_calibration is not None
                else "qcal_inverse_p_error_given_q"
            ),
            "joint_phred_calibration": (
                str(source_phred_path) if joint_phred_calibration is not None else None
            ),
            "recovered_train_phred_calibration": str(recovered_train_phred_path),
            "recovered_test_phred_calibration": str(recovered_test_phred_path),
            "covered_train_truth_phred_calibration": str(covered_train_phred_path),
            "covered_test_truth_phred_calibration": str(covered_test_phred_path),
            "phred_calibration_metrics": str(phred_metrics_path),
            "source_quality_calibration_model": (
                str(source_qcal_path)
                if args.quality_calibration_model_json is not None
                else None
            ),
            "recovered_train_quality_calibration_model": str(recovered_train_qcal_path),
            "recovered_test_quality_calibration_model": str(recovered_test_qcal_path),
            "covered_train_truth_quality_calibration_model": str(covered_train_qcal_path),
            "covered_test_truth_quality_calibration_model": str(covered_test_qcal_path),
            "quality_calibration_model_metrics": str(qcal_metrics_path),
        }
    )
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    logger.info("Wrote retrained model to %s", retrained_model)
    logger.info("Wrote recovery metrics to %s", metrics_path)
    logger.info("Wrote rate recovery metrics to %s", rate_metrics_path)
    logger.info("Wrote Skiver/BAM error match metrics to %s", skiver_bam_match_path)
    logger.info("Wrote VI uncertainty metrics to %s", vi_metrics_path)
    logger.info("Wrote Phred calibration metrics to %s", phred_metrics_path)
    logger.info("Wrote quality calibration model metrics to %s", qcal_metrics_path)
    logger.info(
        "Recovery TV distances: observed/source=%.6f retrained/source=%.6f observed/retrained=%.6f",
        metrics["observed_vs_source_tv"],
        metrics["retrained_vs_source_tv"],
        metrics["observed_vs_retrained_tv"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
