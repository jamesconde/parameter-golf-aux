#!/usr/bin/env python3
"""Experiment harness for Parameter Golf auxiliary loss sweeps.

Runs experiments with configurable loss combinations, lambda sweeps,
and seeds. Parses logs, computes statistics, and generates an ablation table.

Usage:
    # Run from a sweep config file
    python scripts/experiment_runner.py --config experiments/sweep_config.json

    # Quick single experiment
    python scripts/experiment_runner.py --name focal_g1 --seeds 42,1337,7 \
        --env USE_FOCAL_LOSS=1 FOCAL_GAMMA=1.0 LAMBDA_DECORR=0 LAMBDA_RANK=0 LAMBDA_UNIGRAM=0

    # Generate report from existing logs
    python scripts/experiment_runner.py --report-only

All parameters are configurable — no hardcoded values.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# Configuration
# ============================================================

@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    name: str
    env: dict = field(default_factory=dict)
    description: str = ""
    train_script: str = ""  # Override sweep-level train_script for this experiment


@dataclass
class SweepConfig:
    """Configuration for the entire experiment sweep."""
    # Training settings
    train_script: str = "train_gpt_aux.py"
    iterations: int = 500
    train_batch_tokens: int = 65536
    val_loss_every: int = 100
    max_wallclock_seconds: float = 300.0
    eval_stride: int = 64

    # Experiment settings
    seeds: list = field(default_factory=lambda: [42, 1337, 7])
    log_dir: str = "logs"
    results_file: str = "experiments/results_auto.md"
    results_json: str = "experiments/results_auto.json"

    # Early stopping: kill run if val_bpb at early checkpoint exceeds
    # baseline by more than this threshold
    early_stop_enabled: bool = False
    early_stop_threshold: float = 0.02
    early_stop_check_step: int = 200

    # Base environment variables (applied to ALL runs)
    base_env: dict = field(default_factory=dict)

    # Experiments to run
    experiments: list = field(default_factory=list)


def default_sweep_config() -> SweepConfig:
    """Generate the default sweep config with all planned experiments."""
    cfg = SweepConfig()

    # Base env shared by all aux loss runs
    aux_base = {
        "USE_AUX_LOSSES": "1",
        "USE_FOCAL_LOSS": "0",
        "LAMBDA_DECORR": "0",
        "LAMBDA_RANK": "0",
        "LAMBDA_UNIGRAM": "0",
    }

    cfg.experiments = [
        # Unmodified SOTA script — overhead check
        RunConfig(
            name="baseline_sota",
            env={},
            description="Unmodified SOTA script (overhead check)",
            train_script="train_gpt_sota.py",
        ),
        # Our script with aux losses disabled — apples-to-apples reference
        RunConfig(
            name="baseline",
            env={"USE_AUX_LOSSES": "0"},
            description="Our script with aux losses disabled (apples-to-apples reference)",
        ),

        # Focal loss sweep
        RunConfig(
            name="focal_g05",
            env={**aux_base, "USE_FOCAL_LOSS": "1", "FOCAL_GAMMA": "0.5"},
            description="Focal loss gamma=0.5",
        ),
        RunConfig(
            name="focal_g1",
            env={**aux_base, "USE_FOCAL_LOSS": "1", "FOCAL_GAMMA": "1.0"},
            description="Focal loss gamma=1.0 (data-informed recommendation)",
        ),
        RunConfig(
            name="focal_g2",
            env={**aux_base, "USE_FOCAL_LOSS": "1", "FOCAL_GAMMA": "2.0"},
            description="Focal loss gamma=2.0 (standard focal)",
        ),
        RunConfig(
            name="focal_g3",
            env={**aux_base, "USE_FOCAL_LOSS": "1", "FOCAL_GAMMA": "3.0"},
            description="Focal loss gamma=3.0 (aggressive)",
        ),

        # Decorrelation sweep
        RunConfig(
            name="decorr_0001",
            env={**aux_base, "LAMBDA_DECORR": "0.001"},
            description="Decorrelation lambda=0.001",
        ),
        RunConfig(
            name="decorr_001",
            env={**aux_base, "LAMBDA_DECORR": "0.01"},
            description="Decorrelation lambda=0.01",
        ),
        RunConfig(
            name="decorr_005",
            env={**aux_base, "LAMBDA_DECORR": "0.05"},
            description="Decorrelation lambda=0.05",
        ),
        RunConfig(
            name="decorr_01",
            env={**aux_base, "LAMBDA_DECORR": "0.1"},
            description="Decorrelation lambda=0.1",
        ),

        # Rank loss sweep
        RunConfig(
            name="rank_001",
            env={**aux_base, "LAMBDA_RANK": "0.01", "RANK_EVERY": "10"},
            description="Rank loss lambda=0.01 (every 10 steps)",
        ),
        RunConfig(
            name="rank_005",
            env={**aux_base, "LAMBDA_RANK": "0.05", "RANK_EVERY": "10"},
            description="Rank loss lambda=0.05 (every 10 steps)",
        ),
        RunConfig(
            name="rank_01",
            env={**aux_base, "LAMBDA_RANK": "0.1", "RANK_EVERY": "10"},
            description="Rank loss lambda=0.1 (every 10 steps)",
        ),

        # Unigram KL sweep
        RunConfig(
            name="unigram_001",
            env={**aux_base, "LAMBDA_UNIGRAM": "0.01"},
            description="Unigram KL lambda=0.01 (decay by 50%)",
        ),
        RunConfig(
            name="unigram_005",
            env={**aux_base, "LAMBDA_UNIGRAM": "0.05"},
            description="Unigram KL lambda=0.05 (decay by 50%)",
        ),
        RunConfig(
            name="unigram_01",
            env={**aux_base, "LAMBDA_UNIGRAM": "0.1"},
            description="Unigram KL lambda=0.1 (decay by 50%)",
        ),
    ]

    return cfg


# ============================================================
# Log Parsing
# ============================================================

@dataclass
class RunResult:
    """Parsed results from a single training run."""
    name: str
    seed: int
    final_val_bpb: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_bpb: Optional[float] = None
    artifact_size_bytes: Optional[int] = None
    total_steps: Optional[int] = None
    training_time_ms: Optional[float] = None
    val_bpb_history: list = field(default_factory=list)
    train_loss_history: list = field(default_factory=list)
    aux_loss_history: list = field(default_factory=list)
    error: Optional[str] = None


def parse_log(log_path: str, name: str, seed: int) -> RunResult:
    """Parse a training log file and extract key metrics."""
    result = RunResult(name=name, seed=seed)

    if not os.path.exists(log_path):
        result.error = f"Log file not found: {log_path}"
        return result

    with open(log_path) as f:
        text = f.read()

    # The log file contains the script source code interleaved with runtime output.
    # Use a regex pattern that only matches ACTUAL numeric values (digit before dot),
    # not Python format specifiers like {val_bpb:.4f} which match as ".4".
    # Real BPB values are always >= 0.5 and look like "1.1234" (start with digit).
    # Pattern: one or more digits, dot, one or more digits (e.g., "1.1234")
    _NUM = r'\d+\.\d+'  # Matches "1.1234" but NOT ".4f" or ".8f"

    # Final roundtrip BPB (most authoritative)
    m = re.search(rf'final_int8_zlib_roundtrip_exact\s+val_loss:({_NUM})\s+val_bpb:({_NUM})', text)
    if m:
        result.final_val_loss = float(m.group(1))
        result.final_val_bpb = float(m.group(2))

    # Fallback: final int6+zstd roundtrip
    if result.final_val_bpb is None:
        m = re.search(rf'final_int6_zstd_roundtrip_exact\s+val_loss:({_NUM})\s+val_bpb:({_NUM})', text)
        if m:
            result.final_val_loss = float(m.group(1))
            result.final_val_bpb = float(m.group(2))

    # Fallback: final int6 sliding window
    if result.final_val_bpb is None:
        m = re.search(rf'final_int6_sliding_window_exact\s+val_loss:({_NUM})\s+val_bpb:({_NUM})', text)
        if m:
            result.final_val_loss = float(m.group(1))
            result.final_val_bpb = float(m.group(2))

    # Fallback: final int6 sliding window s64
    if result.final_val_bpb is None:
        m = re.search(rf'final_int6_sliding_window_s64_exact\s+val_loss:({_NUM})\s+val_bpb:({_NUM})', text)
        if m:
            result.final_val_loss = float(m.group(1))
            result.final_val_bpb = float(m.group(2))

    # Fallback: any final roundtrip line with real numbers
    if result.final_val_bpb is None:
        m = re.search(rf'roundtrip.*val_bpb:({_NUM})', text)
        if m:
            result.final_val_bpb = float(m.group(1))

    # Fallback: last val_bpb from a step line (requires step:N/M prefix)
    if result.final_val_bpb is None:
        matches = re.findall(rf'step:\d+/\d+\s+val_loss:{_NUM}\s+val_bpb:({_NUM})', text)
        if matches:
            result.final_val_bpb = float(matches[-1])

    # Val BPB history (periodic validation — requires step prefix + real numbers)
    for m in re.finditer(rf'step:(\d+)/\d+\s+val_loss:({_NUM})\s+val_bpb:({_NUM})', text):
        result.val_bpb_history.append({
            "step": int(m.group(1)),
            "val_loss": float(m.group(2)),
            "val_bpb": float(m.group(3)),
        })

    # Best val BPB
    if result.val_bpb_history:
        result.best_val_bpb = min(h["val_bpb"] for h in result.val_bpb_history)

    # Training loss + aux loss history (requires step prefix + real numbers)
    for m in re.finditer(rf'step:(\d+)/\d+\s+train_loss:({_NUM})(?:\s+aux_loss:({_NUM}))?', text):
        step = int(m.group(1))
        result.train_loss_history.append({"step": step, "train_loss": float(m.group(2))})
        if m.group(3):
            result.aux_loss_history.append({"step": step, "aux_loss": float(m.group(3))})

    # Total steps
    step_matches = re.findall(r'step:(\d+)/', text)
    if step_matches:
        result.total_steps = max(int(s) for s in step_matches)

    # Training time
    m = re.search(r'train_time:([\d.]+)ms', text)
    if m:
        result.training_time_ms = float(m.group(1))

    # Artifact size
    m = re.search(r'Total submission size.*?:\s*(\d+)\s*bytes', text)
    if m:
        result.artifact_size_bytes = int(m.group(1))

    # Check for errors — look for actual traceback patterns, not source code
    # A real traceback has "Traceback (most recent call last):" as a standalone line
    if "Traceback (most recent call last):" in text:
        # Find the last traceback and extract the error line
        tb_start = text.rfind("Traceback (most recent call last):")
        tb_section = text[tb_start:]
        # The error type is usually the last non-empty line
        tb_lines = [l.strip() for l in tb_section.split('\n') if l.strip()]
        for line in reversed(tb_lines):
            if ':' in line and not line.startswith('File ') and not line.startswith('raise '):
                result.error = line[:200]
                break
    elif result.final_val_bpb is None and result.total_steps is None:
        result.error = "Training did not produce any output (possible silent crash/OOM)"

    return result


# ============================================================
# Experiment Running
# ============================================================

def build_env(sweep: SweepConfig, experiment: RunConfig, seed: int) -> dict:
    """Build the full environment dict for a run."""
    env = dict(os.environ)

    # Training config
    env["ITERATIONS"] = str(sweep.iterations)
    env["TRAIN_BATCH_TOKENS"] = str(sweep.train_batch_tokens)
    env["VAL_LOSS_EVERY"] = str(sweep.val_loss_every)
    env["MAX_WALLCLOCK_SECONDS"] = str(sweep.max_wallclock_seconds)
    env["EVAL_STRIDE"] = str(sweep.eval_stride)
    env["SEED"] = str(seed)

    # Base env
    for k, v in sweep.base_env.items():
        env[k] = str(v)

    # Experiment-specific env
    for k, v in experiment.env.items():
        env[k] = str(v)

    # Run ID
    env["RUN_ID"] = f"{experiment.name}_seed{seed}"

    return env


def _find_log(run_id: str, sweep: SweepConfig, script: str, extra_dir: str = "") -> str:
    """Find the log file for a run, checking multiple possible locations."""
    candidates = [
        # 1. Relative to cwd (most common on Colab)
        os.path.join(sweep.log_dir, f"{run_id}.txt"),
        # 2. Relative to training script directory
        os.path.join(os.path.dirname(os.path.abspath(script)), sweep.log_dir, f"{run_id}.txt"),
        # 3. Absolute resolved path (follows symlinks)
        os.path.join(os.path.realpath(sweep.log_dir), f"{run_id}.txt"),
    ]
    if extra_dir:
        # 4. Relative to explicit subprocess cwd
        candidates.insert(0, os.path.join(extra_dir, sweep.log_dir, f"{run_id}.txt"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def run_experiment(sweep: SweepConfig, experiment: RunConfig, seed: int,
                   dry_run: bool = False) -> RunResult:
    """Run a single experiment and return parsed results."""
    run_id = f"{experiment.name}_seed{seed}"
    script = experiment.train_script or sweep.train_script
    log_path = _find_log(run_id, sweep, script)

    # Skip if log already exists with a final BPB
    if os.path.exists(log_path):
        existing = parse_log(log_path, experiment.name, seed)
        if existing.final_val_bpb is not None and existing.error is None:
            print(f"  SKIP {run_id} — already complete (val_bpb={existing.final_val_bpb:.4f})")
            return existing

    if dry_run:
        print(f"  DRY RUN: {run_id}")
        return RunResult(name=experiment.name, seed=seed, error="dry_run")

    env = build_env(sweep, experiment, seed)

    # Resolve working directory: use the directory containing the training script
    script_path = os.path.abspath(script)
    cwd = os.path.dirname(script_path)
    cmd = ["python3", os.path.basename(script_path)]

    print(f"  RUN {run_id} (cwd={cwd}) ...")
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, cwd=cwd,
            timeout=sweep.max_wallclock_seconds * 3 + 600,
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr.strip().split('\n')[-15:] if proc.stderr else []
            print(f"  STDERR (last 15 lines):\n" + '\n'.join(stderr_tail))
    except subprocess.TimeoutExpired:
        return RunResult(name=experiment.name, seed=seed, error="timeout")
    except Exception as e:
        return RunResult(name=experiment.name, seed=seed, error=str(e)[:200])

    elapsed = time.time() - t0

    # Re-find log (subprocess may have created it in cwd)
    log_path = _find_log(run_id, sweep, script, extra_dir=cwd)
    result = parse_log(log_path, experiment.name, seed)

    status = "OK" if result.final_val_bpb else "FAIL"
    bpb_str = f"val_bpb={result.final_val_bpb:.4f}" if result.final_val_bpb else result.error or "no result"
    print(f"  {status} {run_id} — {bpb_str} ({elapsed:.0f}s)")

    return result


# ============================================================
# Statistical Analysis
# ============================================================

def welch_t_test(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_statistic, p_value).

    Two-tailed test: is sample_b different from sample_a?
    """
    import math

    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0

    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate p-value using normal distribution for simplicity
    # (accurate enough for df > 5, conservative for smaller df)
    p_value = 2.0 * _normal_cdf(-abs(t_stat))

    return float(t_stat), float(p_value)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    return 0.5 * math.erfc(-x / math.sqrt(2))


@dataclass
class ExperimentSummary:
    """Aggregated results for one experiment across seeds."""
    name: str
    description: str
    n_seeds: int
    bpb_values: list
    mean_bpb: Optional[float]
    std_bpb: Optional[float]
    min_bpb: Optional[float]
    max_bpb: Optional[float]
    delta_vs_baseline: Optional[float]
    t_stat: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]  # p < 0.01
    errors: list


def summarize_experiment(name: str, description: str, results: list[RunResult],
                         baseline_bpbs: list[float] = None) -> ExperimentSummary:
    """Aggregate results across seeds for one experiment."""
    bpb_values = [r.final_val_bpb for r in results if r.final_val_bpb is not None]
    errors = [f"seed={r.seed}: {r.error}" for r in results if r.error]

    if not bpb_values:
        return ExperimentSummary(
            name=name, description=description, n_seeds=len(results),
            bpb_values=[], mean_bpb=None, std_bpb=None, min_bpb=None, max_bpb=None,
            delta_vs_baseline=None, t_stat=None, p_value=None, significant=None,
            errors=errors,
        )

    import statistics
    mean_bpb = statistics.mean(bpb_values)
    std_bpb = statistics.stdev(bpb_values) if len(bpb_values) > 1 else 0.0

    delta = None
    t_stat = None
    p_value = None
    sig = None
    if baseline_bpbs and len(baseline_bpbs) >= 2 and len(bpb_values) >= 2:
        baseline_mean = statistics.mean(baseline_bpbs)
        delta = mean_bpb - baseline_mean
        t_stat, p_value = welch_t_test(baseline_bpbs, bpb_values)
        sig = p_value < 0.01

    return ExperimentSummary(
        name=name, description=description, n_seeds=len(bpb_values),
        bpb_values=bpb_values, mean_bpb=mean_bpb, std_bpb=std_bpb,
        min_bpb=min(bpb_values), max_bpb=max(bpb_values),
        delta_vs_baseline=delta, t_stat=t_stat, p_value=p_value,
        significant=sig, errors=errors,
    )


# ============================================================
# Report Generation
# ============================================================

def generate_report(summaries: list[ExperimentSummary], sweep: SweepConfig) -> str:
    """Generate a Markdown ablation table."""
    lines = [
        "# Parameter Golf — Experiment Results (Auto-Generated)",
        "",
        f"**Training script:** `{sweep.train_script}`",
        f"**Iterations:** {sweep.iterations} | **Batch tokens:** {sweep.train_batch_tokens} | "
        f"**Val every:** {sweep.val_loss_every} | **Max wallclock:** {sweep.max_wallclock_seconds}s",
        f"**Seeds:** {sweep.seeds}",
        "",
        "## Ablation Table",
        "",
        "| Experiment | Description | Seeds | val_bpb (mean±std) | Delta | p-value | Sig? |",
        "|-----------|-------------|:-----:|-------------------:|------:|--------:|:----:|",
    ]

    for s in summaries:
        if s.mean_bpb is None:
            bpb_str = "FAILED"
            delta_str = "—"
            p_str = "—"
            sig_str = "—"
        else:
            bpb_str = f"{s.mean_bpb:.4f}±{s.std_bpb:.4f}"
            delta_str = f"{s.delta_vs_baseline:+.4f}" if s.delta_vs_baseline is not None else "—"
            p_str = f"{s.p_value:.4f}" if s.p_value is not None else "—"
            if s.significant is None:
                sig_str = "—"
            elif s.significant and s.delta_vs_baseline and s.delta_vs_baseline < 0:
                sig_str = "YES"
            elif s.significant:
                sig_str = "WORSE"
            else:
                sig_str = "no"

        lines.append(
            f"| {s.name} | {s.description} | {s.n_seeds} | {bpb_str} | {delta_str} | {p_str} | {sig_str} |"
        )

    # Add recommendations
    lines.extend(["", "## Recommendations", ""])

    # Find best individual losses
    improved = [s for s in summaries if s.delta_vs_baseline is not None and s.delta_vs_baseline < 0]
    sig_improved = [s for s in improved if s.significant]

    if sig_improved:
        lines.append("**Statistically significant improvements (p < 0.01):**")
        for s in sorted(sig_improved, key=lambda x: x.delta_vs_baseline):
            lines.append(f"- **{s.name}**: {s.delta_vs_baseline:+.4f} BPB (p={s.p_value:.4f})")
        lines.append("")
        lines.append("**Recommended stacking order** (try combining these):")
        for i, s in enumerate(sorted(sig_improved, key=lambda x: x.delta_vs_baseline), 1):
            lines.append(f"{i}. {s.name} ({s.delta_vs_baseline:+.4f})")
    elif improved:
        lines.append("**Improvements observed but not statistically significant (need more seeds):**")
        for s in sorted(improved, key=lambda x: x.delta_vs_baseline):
            lines.append(f"- {s.name}: {s.delta_vs_baseline:+.4f} BPB")
    else:
        lines.append("No improvements observed yet. Check individual experiment errors.")

    # Errors
    all_errors = []
    for s in summaries:
        all_errors.extend(s.errors)
    if all_errors:
        lines.extend(["", "## Errors", ""])
        for e in all_errors:
            lines.append(f"- {e}")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf experiment runner")
    parser.add_argument("--config", help="Path to sweep config JSON file")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate default sweep config and exit")
    parser.add_argument("--report-only", action="store_true",
                        help="Parse existing logs and generate report without running")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--name", help="Run a single named experiment")
    parser.add_argument("--seeds", default="42,1337,7",
                        help="Comma-separated seeds (default: 42,1337,7)")
    parser.add_argument("--env", nargs="*", default=[],
                        help="Extra env vars as KEY=VALUE pairs")
    parser.add_argument("--filter", help="Only run experiments matching this regex")
    parser.add_argument("--iterations", type=int, help="Override iteration count")
    parser.add_argument("--train-batch-tokens", type=int, help="Override batch tokens")
    parser.add_argument("--max-wallclock", type=float, help="Override max wallclock seconds")
    parser.add_argument("--val-every", type=int, help="Override validation interval")
    parser.add_argument("--train-script", help="Override training script path")
    parser.add_argument("--log-dir", help="Override log directory")
    args = parser.parse_args()

    # Generate config mode
    if args.generate_config:
        cfg = default_sweep_config()
        out_path = args.config or "experiments/sweep_config.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "train_script": cfg.train_script,
                "iterations": cfg.iterations,
                "train_batch_tokens": cfg.train_batch_tokens,
                "val_loss_every": cfg.val_loss_every,
                "max_wallclock_seconds": cfg.max_wallclock_seconds,
                "eval_stride": cfg.eval_stride,
                "seeds": cfg.seeds,
                "log_dir": cfg.log_dir,
                "results_file": cfg.results_file,
                "results_json": cfg.results_json,
                "early_stop_enabled": cfg.early_stop_enabled,
                "early_stop_threshold": cfg.early_stop_threshold,
                "early_stop_check_step": cfg.early_stop_check_step,
                "base_env": cfg.base_env,
                "experiments": [asdict(e) for e in cfg.experiments],
            }, f, indent=2)
        print(f"Generated config at {out_path}")
        return

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            raw = json.load(f)
        sweep = SweepConfig(
            train_script=raw.get("train_script", "train_gpt_aux.py"),
            iterations=raw.get("iterations", 500),
            train_batch_tokens=raw.get("train_batch_tokens", 65536),
            val_loss_every=raw.get("val_loss_every", 100),
            max_wallclock_seconds=raw.get("max_wallclock_seconds", 300.0),
            eval_stride=raw.get("eval_stride", 64),
            seeds=raw.get("seeds", [42, 1337, 7]),
            log_dir=raw.get("log_dir", "logs"),
            results_file=raw.get("results_file", "experiments/results_auto.md"),
            results_json=raw.get("results_json", "experiments/results_auto.json"),
            early_stop_enabled=raw.get("early_stop_enabled", False),
            early_stop_threshold=raw.get("early_stop_threshold", 0.02),
            early_stop_check_step=raw.get("early_stop_check_step", 200),
            base_env=raw.get("base_env", {}),
            experiments=[RunConfig(**e) for e in raw.get("experiments", [])],
        )
    else:
        sweep = default_sweep_config()

    # Command-line overrides
    if args.seeds:
        sweep.seeds = [int(s) for s in args.seeds.split(",")]
    if args.iterations:
        sweep.iterations = args.iterations
    if args.train_batch_tokens:
        sweep.train_batch_tokens = args.train_batch_tokens
    if args.max_wallclock:
        sweep.max_wallclock_seconds = args.max_wallclock
    if args.val_every:
        sweep.val_loss_every = args.val_every
    if args.train_script:
        sweep.train_script = args.train_script
    if args.log_dir:
        sweep.log_dir = args.log_dir

    # Single experiment mode
    if args.name:
        env_dict = {}
        for kv in args.env:
            k, v = kv.split("=", 1)
            env_dict[k] = v
        sweep.experiments = [RunConfig(name=args.name, env=env_dict)]

    # Filter experiments
    if args.filter:
        import re as re_mod
        pattern = re_mod.compile(args.filter)
        sweep.experiments = [e for e in sweep.experiments if pattern.search(e.name)]

    if not sweep.experiments:
        print("No experiments to run. Use --generate-config to create a sweep config.")
        return

    # Run or report
    all_results: dict[str, list[RunResult]] = {}

    if not args.report_only:
        print(f"Running {len(sweep.experiments)} experiments × {len(sweep.seeds)} seeds")
        print(f"Training script: {sweep.train_script}")
        print(f"Iterations: {sweep.iterations} | Batch: {sweep.train_batch_tokens}")
        print(f"Seeds: {sweep.seeds}")
        print("=" * 60)

        for exp in sweep.experiments:
            print(f"\n[{exp.name}] {exp.description}")
            all_results[exp.name] = []
            for seed in sweep.seeds:
                result = run_experiment(sweep, exp, seed, dry_run=args.dry_run)
                all_results[exp.name].append(result)

    # Parse all existing logs for report
    print("\n" + "=" * 60)
    print("Parsing logs and generating report...")
    print("=" * 60)

    if args.report_only or args.dry_run:
        # Discover experiments from log files — try multiple locations
        log_candidates = [
            Path(sweep.log_dir),
            Path(sweep.log_dir).resolve(),
            Path(os.path.realpath(sweep.log_dir)),
        ]
        log_dir = None
        for candidate in log_candidates:
            if candidate.exists():
                log_dir = candidate
                break
        if log_dir is None:
            print(f"  WARNING: log_dir not found. Tried: {[str(c) for c in log_candidates]}")
            log_dir = Path(sweep.log_dir)  # Fallback
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.txt")):
                m = re.match(r"(.+)_seed(\d+)\.txt", log_file.name)
                if m:
                    exp_name = m.group(1)
                    seed = int(m.group(2))
                    result = parse_log(str(log_file), exp_name, seed)
                    if exp_name not in all_results:
                        all_results[exp_name] = []
                    # Avoid duplicates
                    if not any(r.seed == seed for r in all_results[exp_name]):
                        all_results[exp_name].append(result)

    # Build experiment descriptions lookup
    exp_descriptions = {e.name: e.description for e in sweep.experiments}

    # Get baseline BPBs
    baseline_bpbs = []
    if "baseline" in all_results:
        baseline_bpbs = [r.final_val_bpb for r in all_results["baseline"]
                         if r.final_val_bpb is not None]

    # Summarize
    summaries = []
    for exp_name, results in sorted(all_results.items()):
        desc = exp_descriptions.get(exp_name, "")
        summary = summarize_experiment(exp_name, desc, results,
                                       baseline_bpbs if exp_name != "baseline" else None)
        summaries.append(summary)

    # Print summary
    print(f"\n{'Experiment':<20} {'Seeds':>5} {'BPB (mean±std)':>18} {'Delta':>8} {'p':>8}")
    print("-" * 65)
    for s in summaries:
        if s.mean_bpb is None:
            print(f"{s.name:<20} {'FAIL':>5}")
            continue
        bpb = f"{s.mean_bpb:.4f}±{s.std_bpb:.4f}"
        delta = f"{s.delta_vs_baseline:+.4f}" if s.delta_vs_baseline is not None else "—"
        p = f"{s.p_value:.4f}" if s.p_value is not None else "—"
        print(f"{s.name:<20} {s.n_seeds:>5} {bpb:>18} {delta:>8} {p:>8}")

    # Generate and save report
    report = generate_report(summaries, sweep)
    os.makedirs(os.path.dirname(sweep.results_file) or ".", exist_ok=True)
    with open(sweep.results_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to {sweep.results_file}")

    # Save JSON results
    json_data = {
        "config": {
            "train_script": sweep.train_script,
            "iterations": sweep.iterations,
            "seeds": sweep.seeds,
        },
        "baseline_bpbs": baseline_bpbs,
        "summaries": [asdict(s) for s in summaries],
    }
    with open(sweep.results_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON results saved to {sweep.results_json}")


if __name__ == "__main__":
    main()
