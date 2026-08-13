import argparse
import os
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
    
    start_idx = 0
    mid_idx = length // 2
    end_idx = length - 1
    return [start_idx, mid_idx, end_idx]

def process_and_plot(csv1_path, pid1, csv2_path, pid2, output_img="fault_comparison.png"):
    # 1. Load CSV files
    print(f"[+] Loading {csv1_path}...")
    df1 = pd.read_csv(csv1_path)
    
    print(f"[+] Loading {csv2_path}...")
    df2 = pd.read_csv(csv2_path)

    # Clean whitespace in column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # 2. Filter for specific PID and Processor Id == 1
    filtered_df1 = df1[(df1['PID'] == pid1) & (df1['Processor Id'] == 1)].sort_values('Iter').reset_index(drop=True)
    filtered_df2 = df2[(df2['PID'] == pid2) & (df2['Processor Id'] == 1)].sort_values('Iter').reset_index(drop=True)

    if filtered_df1.empty:
        print(f"[!] Warning: No data found for PID {pid1} with Processor Id 1 in {csv1_path}")
    if filtered_df2.empty:
        print(f"[!] Warning: No data found for PID {pid2} with Processor Id 1 in {csv2_path}")

    # 3. Screen-Friendly Figure Size (8x5 inches fits comfortably on almost all displays)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)  # Screen-friendly dimensions & DPI

    # Color Palette
    color_p1 = '#1f77b4'  # Professional Dark Blue
    color_p2 = '#d62728'  # Crimson Red

    # Plot Process 1 (Thin lines: linewidth=1.2, smaller markers: markersize=4)
    if not filtered_df1.empty:
        ax.plot(
            filtered_df1['Iter'], 
            filtered_df1['Number of Faults'], 
            marker='o', 
            markersize=4,
            linestyle='-', 
            linewidth=1.2,  # Thinner line reveals small fluctuations
            color=color_p1,
            label=f"Process 1 (PID: {pid1})"
        )

        # Annotate only Start, Middle, and End points for Process 1
        key_indices_1 = get_key_indices(len(filtered_df1))
        for idx in key_indices_1:
            row = filtered_df1.iloc[idx]
            x, y = row['Iter'], row['Number of Faults']
            ax.annotate(
                f"({int(x)}s, {int(y):,})",
                xy=(x, y),
                xytext=(0, 8),  # Shift upwards
                textcoords="offset points",
                ha='center',
                fontsize=7.5,
                fontweight='bold',
                color=color_p1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_p1, lw=0.6, alpha=0.9)
            )

    # Plot Process 2 (Thin lines: linewidth=1.2, smaller markers: markersize=4)
    if not filtered_df2.empty:
        ax.plot(
            filtered_df2['Iter'], 
            filtered_df2['Number of Faults'], 
            marker='s', 
            markersize=4,
            linestyle='--', 
            linewidth=1.2,  # Thinner line reveals small fluctuations
            color=color_p2,
            label=f"Process 2 (PID: {pid2})"
        )

        # Annotate only Start, Middle, and End points for Process 2
        key_indices_2 = get_key_indices(len(filtered_df2))
        for idx in key_indices_2:
            row = filtered_df2.iloc[idx]
            x, y = row['Iter'], row['Number of Faults']
            ax.annotate(
                f"({int(x)}s, {int(y):,})",
                xy=(x, y),
                xytext=(0, -15),  # Shift downwards to avoid overlap
                textcoords="offset points",
                ha='center',
                fontsize=7.5,
                fontweight='bold',
                color=color_p2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_p2, lw=0.6, alpha=0.9)
            )

    # 4. Y-Axis Formatting (Real integer values with commas)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 5. Titles, Labels, and Legend
    ax.set_xlabel('Time / Iteration (Seconds)', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_ylabel('Number of Page Faults', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_title('UVM Page Fault Benchmark Comparison (Processor ID 1)', fontsize=11, fontweight='bold', pad=10)

    # Grid and Border Aesthetics
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Legend
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=9, loc='best')

    plt.tight_layout()

    # 6. Save high-resolution PNG on disk, but show compact interactive window on screen
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] High-resolution plot saved to disk: {output_img}")
    
    # Show figure window scaled nicely on desktop displays
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare UVM page faults across two CSV files.")
    parser.add_argument("--csv1", type=str, default="run1.csv", help="Path to first CSV file")
    parser.add_argument("--pid1", type=int, default=115050, help="PID for first CSV file")
    parser.add_argument("--csv2", type=str, default="run2.csv", help="Path to second CSV file")
    parser.add_argument("--pid2", type=int, default=115042, help="PID for second CSV file")
    parser.add_argument("--out", type=str, default="fault_comparison.png", help="Output image file name")

    args = parser.parse_args()

    process_and_plot(args.csv1, args.pid1, args.csv2, args.pid2, args.out)