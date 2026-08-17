import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_key_indices(length):
    """Returns indices for start, middle, and end points of a dataset."""
    if length <= 0:
        return []
    if length == 1:
        return [0]
    if length == 2:
        return [0, 1]
    
    return [0, length // 2, length - 1]

def process_and_plot(csv1, pid1, csv2, pid2, csv3, pid3, output_img="resident_pages_comparison_3way.png"):
    # 1. Load CSVs
    print(f"[+] Loading {csv1}...")
    df1 = pd.read_csv(csv1)
    print(f"[+] Loading {csv2}...")
    df2 = pd.read_csv(csv2)
    print(f"[+] Loading {csv3}...")
    df3 = pd.read_csv(csv3)

    # Clean header whitespace
    for df in [df1, df2, df3]:
        df.columns = df.columns.str.strip()

    # Target column to compare
    target_col = 'Resident Pages'

    # Verify column exists
    for idx, df in enumerate([df1, df2, df3], start=1):
        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found in CSV {idx}. Available columns: {list(df.columns)}")

    # 2. Filter data for Processor Id == 1
    f1 = df1[(df1['PID'] == pid1) & (df1['Processor Id'] == 1)].sort_values('Iter').reset_index(drop=True)
    f2 = df2[(df2['PID'] == pid2) & (df2['Processor Id'] == 1)].sort_values('Iter').reset_index(drop=True)
    f3 = df3[(df3['PID'] == pid3) & (df3['Processor Id'] == 1)].sort_values('Iter').reset_index(drop=True)

    # 3. Figure Setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    # Visual configurations for the 3 runs
    series_config = [
        (f1, pid1, '#1f77b4', '-', 'o', 8, f"CSV 1 (PID: {pid1})"),     # Solid Blue
        (f2, pid2, '#d62728', '--', 's', -16, f"CSV 2 (PID: {pid2})"),   # Dashed Red
        (f3, pid3, '#2ca02c', ':', '^', 20, f"CSV 3 (PID: {pid3})")      # Dotted Green
    ]

    # 4. Plot Traces and Annotate Key Points (Start, Middle, End)
    for df, pid, color, ls, marker, text_offset, label in series_config:
        if not df.empty:
            ax.plot(
                df['Iter'], 
                df[target_col], 
                marker=marker, 
                markersize=4,
                linestyle=ls, 
                linewidth=1.2, 
                color=color,
                label=label
            )

            # Annotate key data points
            key_indices = get_key_indices(len(df))
            for idx in key_indices:
                row = df.iloc[idx]
                x, y = row['Iter'], row[target_col]
                ax.annotate(
                    f"({int(x)}s, {int(y):,})",
                    xy=(x, y),
                    xytext=(0, text_offset),
                    textcoords="offset points",
                    ha='center',
                    fontsize=7,
                    fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.9)
                )

    # 5. Axis Formatting
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 6. Labels, Spines, and Legend
    ax.set_xlabel('Time / Iteration (Seconds)', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_ylabel('Resident Pages', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_title('UVM Resident Pages Comparison (Processor ID 1)', fontsize=11, fontweight='bold', pad=10)

    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=8.5, loc='best')

    plt.tight_layout()

    # 7. Save high-resolution PNG
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] High-resolution PNG saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare UVM Resident Pages across 3 CSV files into a static PNG.")
    parser.add_argument("--csv1", type=str, default="run1.csv", help="Path to first CSV")
    parser.add_argument("--pid1", type=int, default=22846, help="PID for first CSV")
    parser.add_argument("--csv2", type=str, default="run2.csv", help="Path to second CSV")
    parser.add_argument("--pid2", type=int, default=115042, help="PID for second CSV")
    parser.add_argument("--csv3", type=str, default="run3.csv", help="Path to third CSV")
    parser.add_argument("--pid3", type=int, default=115060, help="PID for third CSV")
    parser.add_argument("--out", type=str, default="resident_pages_comparison.png", help="Output PNG file name")

    args = parser.parse_args()

    process_and_plot(args.csv1, args.pid1, args.csv2, args.pid2, args.csv3, args.pid3, args.out)