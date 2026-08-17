import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_hard_limits_bar(output_img="hard_limit_page_faults.png"):
    # Data definition
    hard_limits = ['2GB', '4GB', '8GB', '16GB']
    page_faults = [4971078, 4439695, 635329, 630887]

    # Figure Setup (Compact 8x5 layout, 300 DPI for PNG export)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    # Color gradient for visual distinction (darker blue to lighter blue)
    colors = ['#1f77b4', '#3182bd', '#6baed6', '#9ecae1']

    # Draw Bar Chart
    bars = ax.bar(
        hard_limits, 
        page_faults, 
        color=colors, 
        edgecolor='#333333', 
        linewidth=1.0, 
        width=0.55
    )

    # Annotate exact value labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:,}',  # Format numbers with commas (e.g., 4,971,078)
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),  # 6 points vertical offset
            textcoords="offset points",
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#222222'
        )

    # Format Y-axis to show commas (e.g., 1,000,000)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Extend Y-axis limit slightly so text annotations don't get clipped at the top
    max_y = max(page_faults)
    ax.set_ylim(0, max_y * 1.12)

    # Axis Labels & Title
    ax.set_xlabel('Hard Memory Limit', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel('Page Faults', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title('Impact of Memory Hard Limits on Page Faults', fontsize=11, fontweight='bold', pad=12)

    # Grid and Border Aesthetics
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.6)
    ax.set_axisbelow(True)  # Grid lines stay behind the bars
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    plt.tight_layout()

    # Save PNG figure
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Bar graph successfully saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Page Faults vs Memory Hard Limits as a Bar Graph.")
    parser.add_argument("--out", type=str, default="hard_limit_page_faults.png", help="Output PNG file name")
    args = parser.parse_args()

    plot_hard_limits_bar(args.out)