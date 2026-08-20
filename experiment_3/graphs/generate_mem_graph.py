#!/usr/bin/env python3
"""
generate_mem_graph.py

Generates stacked "Physical Memory Allocated vs Time" graphs (matching the
style of Figure 7: stacked bars per process + dashed hard-limit line) from
nuvmtop/BoxD-style CSV logs.

Expected CSV columns (header row required):
    Iter, PID, Processor Id, Number of Faults, Evictions, Resident Pages,
    Physical Memory Allocated, Memory Evicted of other Processes, Thrashed Pages

Rules applied automatically:
    - Rows with Processor Id == 0 are dropped (per spec).
    - Only the 3 PIDs you specify per case are kept.
    - 'Iter' is converted to seconds using --interval (default 0.4s per iter).
    - 'Physical Memory Allocated' is assumed to be in bytes and is converted to GB (GiB, /1024^3).
    - PID order you give per case maps to labels via --ask-sizes (default 8,4,2 GB).

-----------------------------------------------------------------------------
USAGE - one case:

    python3 generate_mem_graph.py \
        --case "16GB Hard Limit:run16.csv:127967,127966,127968:16" \
        --out case_16gb.png

USAGE - all four cases in one 2x2 figure (like the reference figure):

    python3 generate_mem_graph.py \
        --case "16GB Hard Limit:run16.csv:127967,127966,127968:16" \
        --case "14GB Hard Limit:run14.csv:127970,127971,127972:14" \
        --case "10GB Hard Limit:run10.csv:127975,127976,127977:10" \
        --case "7GB Hard Limit:run7.csv:127980,127981,127982:7" \
        --out combined_memory.png

--case format (colon-separated, PIDs comma-separated):
    "<Label>:<csv_path>:<pid_8gb>,<pid_4gb>,<pid_2gb>:<hard_limit_GB>"

Order of PIDs in each --case MUST be: 8GB-asking PID, 4GB-asking PID, 2GB-asking PID
(i.e. exactly the order you described: first PID given = 8GB process, etc.)
-----------------------------------------------------------------------------
"""

import argparse
import math
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BYTES_PER_GB = 1024 ** 3

# Colors roughly matching the reference figure (blue / tan-orange / teal-green)
DEFAULT_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_case(csv_path, pids, interval_s):
    """Load a CSV, filter to Processor Id == 1 and the given PIDs, return a
    time-indexed pivot table (rows=time in seconds, columns=pid) of GB allocated."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Drop the "Processor Id == 0" rows per spec
    df = df[df["Processor Id"] == 1].copy()

    # Keep only the requested PIDs for this case
    df = df[df["PID"].isin(pids)].copy()

    if df.empty:
        raise ValueError(
            f"No rows found in {csv_path} for Processor Id==1 and PIDs {pids}. "
            f"Check that these PIDs actually appear with Processor Id==1 in this file."
        )

    df["Time_s"] = df["Iter"] * interval_s
    df["GB"] = df["Physical Memory Allocated"] / BYTES_PER_GB

    pivot = df.pivot_table(
        index="Time_s", columns="PID", values="GB", aggfunc="sum", fill_value=0
    )

    # Reindex so columns always appear in the exact order the PIDs were given,
    # even if a PID is missing some Iter rows (fill with 0).
    pivot = pivot.reindex(columns=pids, fill_value=0)
    pivot = pivot.sort_index()
    return pivot


def plot_case(ax, pivot, pids, hard_limit_gb, ask_sizes, interval_s,
              colors=DEFAULT_COLORS, title=None, show_legend=False, ymax=None):
    times = pivot.index.values
    bottom = np.zeros(len(times))

    for i, pid in enumerate(pids):
        values = pivot[pid].values
        label = f"{ask_sizes[i]} GB Ask"
        ax.bar(
            times, values, bottom=bottom, width=interval_s,
            color=colors[i % len(colors)], label=label,
            align="center", edgecolor="none",
        )
        bottom += values

    # Hard limit dashed line
    ax.axhline(y=hard_limit_gb, color="black", linestyle="--", linewidth=1.2)
    x_span = times.max() - times.min() if len(times) > 1 else 1
    ax.text(
        times.min() + 0.02 * x_span,
        hard_limit_gb + (ymax or hard_limit_gb * 1.2) * 0.02,
        f"Hard Limit\n({hard_limit_gb:g}GB)",
        fontsize=9, va="bottom", ha="left",
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Physical Memory Allocated (GB)")
    if title:
        ax.set_title(title, fontsize=11)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    if show_legend:
        ax.legend(title="Request Type", loc="upper right", fontsize=8, title_fontsize=9)


def parse_case_arg(raw, ask_sizes):
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"--case must have 4 colon-separated fields "
            f"'Label:csv_path:pid1,pid2,pid3:hard_limit_GB', got: {raw}"
        )
    label, csv_path, pid_str, hard_limit_str = parts
    pids = [int(p.strip()) for p in pid_str.split(",")]
    if len(pids) != len(ask_sizes):
        raise ValueError(
            f"Expected {len(ask_sizes)} PIDs (matching --ask-sizes {ask_sizes}), "
            f"got {len(pids)} in case '{label}'"
        )
    hard_limit_gb = float(hard_limit_str)
    return {"label": label, "csv_path": csv_path, "pids": pids, "hard_limit_gb": hard_limit_gb}


def main():
    ap = argparse.ArgumentParser(
        description="Generate stacked physical-memory-allocated graphs from nuvmtop/BoxD CSV logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--case", action="append", required=True,
        help="Case spec: 'Label:csv_path:pid1,pid2,pid3:hard_limit_GB' (repeatable, up to 4)",
    )
    ap.add_argument(
        "--interval", type=float, default=0.4,
        help="Seconds per Iter (default: 0.4)",
    )
    ap.add_argument(
        "--ask-sizes", type=str, default="8,4,2",
        help="Comma-separated GB sizes matching PID order in --case (default: 8,4,2)",
    )
    ap.add_argument(
        "--out", type=str, default="memory_graph.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI",
    )
    args = ap.parse_args()

    ask_sizes = [s.strip() for s in args.ask_sizes.split(",")]
    cases = [parse_case_arg(c, ask_sizes) for c in args.case]

    # Load all cases first so we can compute a shared y-axis max for comparability
    loaded = []
    max_val = 0.0
    for case in cases:
        pivot = load_case(case["csv_path"], case["pids"], args.interval)
        max_val = max(max_val, pivot.sum(axis=1).max(), case["hard_limit_gb"])
        loaded.append((case, pivot))

    ymax = max_val * 1.2

    n = len(loaded)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (case, pivot) in enumerate(loaded):
        ax = axes_flat[idx]
        plot_case(
            ax, pivot, case["pids"], case["hard_limit_gb"], ask_sizes, args.interval,
            title=case["label"], show_legend=(idx == 0), ymax=ymax,
        )

    # Hide any unused subplot axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()