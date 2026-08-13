import argparse
import pandas as pd
import plotly.graph_objects as go

def process_and_plot_interactive(csv1_path, pid1, csv2_path, pid2, output_html="fault_comparison.html"):
    # 1. Load CSVs
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # 2. Filter data
    f1 = df1[(df1['PID'] == pid1) & (df1['Processor Id'] == 1)].sort_values('Iter')
    f2 = df2[(df2['PID'] == pid2) & (df2['Processor Id'] == 1)].sort_values('Iter')

    # 3. Create Interactive Figure
    fig = go.Figure()

    # Add Process 1
    if not f1.empty:
        fig.add_trace(go.Scatter(
            x=f1['Iter'],
            y=f1['Number of Faults'],
            mode='lines+markers',
            name=f'Process 1 (PID: {pid1})',
            line=dict(width=2, color='#1f77b4'),
            hovertemplate='<b>Time:</b> %{x}s<br><b>Faults:</b> %{y:,.0f}<extra></extra>'
        ))

    # Add Process 2
    if not f2.empty:
        fig.add_trace(go.Scatter(
            x=f2['Iter'],
            y=f2['Number of Faults'],
            mode='lines+markers',
            name=f'Process 2 (PID: {pid2})',
            line=dict(width=2, dash='dash', color='#d62728'),
            hovertemplate='<b>Time:</b> %{x}s<br><b>Faults:</b> %{y:,.0f}<extra></extra>'
        ))

    # 4. Layout & Styling
    fig.update_layout(
        title='UVM Page Fault Benchmark Comparison (Processor ID 1)',
        xaxis_title='Time / Iteration (Seconds)',
        yaxis_title='Number of Page Faults',
        template='plotly_white',
        hovermode='x unified', # Shows comparison hover box at exact time step
        yaxis=dict(tickformat=',d') # Real numbers with commas
    )

    # 5. Save as standalone HTML file
    fig.write_html(output_html)
    print(f"[+] Truly interactive plot saved to: {output_html}")
    
    # Opens automatically in your web browser
    fig.show()

if __name__ == "__main__":
    process_and_plot_interactive("run1.csv", 115050, "run2.csv", 115042)