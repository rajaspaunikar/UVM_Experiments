import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot(csv1_path, pid1, csv2_path, pid2, output_img="fault_comparison.png"):
    # 1. Load CSV files
    print(f"[+] Loading {csv1_path}...")
    df1 = pd.read_csv(csv1_path)
    
    print(f"[+] Loading {csv2_path}...")
    df2 = pd.read_csv(csv2_path)

    # Clean whitespace in column names (e.g., " Number of Faults " -> "Number of Faults")
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # 2. Filter for specific PID and Processor Id == 1
    filtered_df1 = df1[(df1['PID'] == pid1) & (df1['Processor Id'] == 1)].sort_values('Iter')
    filtered_df2 = df2[(df2['PID'] == pid2) & (df2['Processor Id'] == 1)].sort_values('Iter')

    if filtered_df1.empty:
        print(f"[!] Warning: No data found for PID {pid1} with Processor Id 1 in {csv1_path}")
    if filtered_df2.empty:
        print(f"[!] Warning: No data found for PID {pid2} with Processor Id 1 in {csv2_path}")

    # 3. Create the Plot
    plt.figure(figsize=(10, 6))

    # Plot Process 1
    plt.plot(
        filtered_df1['Iter'], 
        filtered_df1['Number of Faults'], 
        marker='o', 
        linestyle='-', 
        linewidth=2, 
        label=f"CSV 1 (PID: {pid1})"
    )

    # Plot Process 2
    plt.plot(
        filtered_df2['Iter'], 
        filtered_df2['Number of Faults'], 
        marker='s', 
        linestyle='--', 
        linewidth=2, 
        label=f"CSV 2 (PID: {pid2})"
    )

    # 4. Styling and Labels
    plt.xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Faults', fontsize=12, fontweight='bold')
    plt.title('UVM Page Fault Comparison (Processor ID 1)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # 5. Save and Show
    plt.savefig(output_img, dpi=300)
    print(f"[+] Plot saved successfully to: {output_img}")
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