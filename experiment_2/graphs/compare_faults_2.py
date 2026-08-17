import argparse
import pandas as pd
import plotly.graph_objects as go

def process_and_plot_interactive(csv1_path, pid1, csv2_path, pid2, csv3_path, pid3, output_html="fault_comparison_3way.html"):
    # 1. Load CSVs
    print(f"[+] Loading CSV 1: {csv1_path}")
    df1 = pd.read_csv(csv1_path)
    
    print(f"[+] Loading CSV 2: {csv2_path}")
    df2 = pd.read_csv(csv2_path)

    print(f"[+] Loading CSV 3: {csv3_path}")
    df3 = pd.read_csv(csv3_path)

    # Clean whitespace in column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()

    # 2. Filter data for Processor Id == 1
    f1 = df1[(df1['PID'] == pid1) & (df1['Processor Id'] == 1)].sort_values('Iter')
    f2 = df2[(df2['PID'] == pid2) & (df2['Processor Id'] == 1)].sort_values('Iter')
    f3 = df3[(df3['PID'] == pid3) & (df3['Processor Id'] == 1)].sort_values('Iter')

    if f1.empty:
        print(f"[!] Warning: No data found for PID {pid1} in {csv1_path}")
    if f2.empty:
        print(f"[!] Warning: No data found for PID {pid2} in {csv2_path}")
    if f3.empty:
        print(f"[!] Warning: No data found for PID {pid3} in {csv3_path}")

    # 3. Create Interactive Figure
    fig = go.Figure()

    # Add Process 1 (Solid Blue)
    if not f1.empty:
        fig.add_trace(go.Scatter(
            x=f1['Iter'],
            y=f1['Number of Faults'],
            mode='lines+markers',
            name=f'Process 1 (PID: {pid1})',
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='<b>Time:</b> %{x}s<br><b>Faults:</b> %{y:,.0f}<extra></extra>'
        ))

    # Add Process 2 (Dashed Red)
    if not f2.empty:
        fig.add_trace(go.Scatter(
            x=f2['Iter'],
            y=f2['Number of Faults'],
            mode='lines+markers',
            name=f'Process 2 (PID: {pid2})',
            line=dict(width=2, dash='dash', color='#d62728'),
            hovertemplate='<b>Time:</b> %{x}s<br><b>Faults:</b> %{y:,.0f}<extra></extra>'
        ))

    # Add Process 3 (Dotted Green)
    if not f3.empty:
        fig.add_trace(go.Scatter(
            x=f3['Iter'],
            y=f3['Number of Faults'],
            mode='lines+markers',
            name=f'Process 3 (PID: {pid3})',
            line=dict(width=2, dash='dot', color='#2ca02c'),
            hovertemplate='<b>Time:</b> %{x}s<br><b>Faults:</b> %{y:,.0f}<extra></extra>'
        ))

    # 4. Layout & Styling
    fig.update_layout(
        title='UVM Page Fault Benchmark Comparison - 3 Runs (Processor ID 1)',
        xaxis_title='Time / Iteration (Seconds)',
        yaxis_title='Number of Page Faults',
        template='plotly_white',
        hovermode='x unified',  # Displays unified hover box comparing all 3 PIDs at that second
        yaxis=dict(tickformat=',d')  # Real numbers formatted with commas
    )

    # 5. Save as standalone HTML file and launch viewer
    fig.write_html(output_html)
    print(f"[+] Interactive plot saved to: {output_html}")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare UVM page faults across three CSV files using Plotly.")
    parser.add_argument("--csv1", type=str, default="run1.csv", help="Path to first CSV file")
    parser.add_argument("--pid1", type=int, default=115050, help="PID for first CSV file")
    parser.add_argument("--csv2", type=str, default="run2.csv", help="Path to second CSV file")
    parser.add_argument("--pid2", type=int, default=115042, help="PID for second CSV file")
    parser.add_argument("--csv3", type=str, default="run3.csv", help="Path to third CSV file")
    parser.add_argument("--pid3", type=int, default=115060, help="PID for third CSV file")
    parser.add_argument("--out", type=str, default="fault_comparison_3way.html", help="Output HTML file name")

    args = parser.parse_args()

    process_and_plot_interactive(
        args.csv1, args.pid1, 
        args.csv2, args.pid2, 
        args.csv3, args.pid3, 
        args.out
    )