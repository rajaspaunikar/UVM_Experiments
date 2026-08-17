import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_memory_limits_comparison(output_img="memory_limits_comparison.png"):
    # Category Labels
    categories = ['2GB', '4GB', '8GB', '16GB']
    
    # Page Fault Data
    hard_limit_faults = [4971078, 4439695, 635329, 630887]
    soft_limit_faults = [None, 631558, 630602, 632734]  # None for 2GB

    # Bar Layout Setup
    x = np.arange(len(categories))
    width = 0.35  # Width of each bar

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)

    # Plot Bars
    rects1 = ax.bar(x - width/2, hard_limit_faults, width, label='Hard Limit', color='#1f77b4', edgecolor='#333333', lw=0.8)
    rects2 = ax.bar(x + width/2, [v if v is not None else 0 for v in soft_limit_faults], width, label='Soft Limit', color='#ff7f0e', edgecolor='#333333', lw=0.8)

    # Function to add numeric labels on top of bars
    def annotate_bars(rects, raw_values):
        for rect, val in zip(rects, raw_values):
            if val is None or val == 0:
                continue
            height = rect.get_height()
            ax.annotate(
                f'{height:,}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                color='#222222'
            )

    annotate_bars(rects1, hard_limit_faults)
    annotate_bars(rects2, soft_limit_faults)

    # Y-axis limits & formatting
    max_y = max(hard_limit_faults)
    ax.set_ylim(0, max_y * 1.12)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Axis Labels, Title, and Legend
    ax.set_xlabel('Memory Limit Configuration', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel('Number of Page Faults', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title('Page Fault Comparison: Hard Limit vs. Soft Limit', fontsize=11, fontweight='bold', pad=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')

    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=9, loc='upper right')

    plt.tight_layout()

    # Save PNG Output
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Comparison graph successfully saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Hard vs Soft Limit Page Faults side-by-side.")
    parser.add_argument("--out", type=str, default="memory_limits_comparison.png", help="Output PNG file name")
    args = parser.parse_args()

    plot_memory_limits_comparison(args.out)