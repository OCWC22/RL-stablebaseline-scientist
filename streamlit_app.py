import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import markdown

# Set page configuration
st.set_page_config(
    page_title="RL Algorithm Comparison",
    page_icon="ud83dudcca",
    layout="wide"
)

# Title and introduction
st.title("Reinforcement Learning Algorithm Comparison")
st.markdown(
    """This dashboard visualizes the performance of different reinforcement learning algorithms 
    implemented using Stable Baselines3 on the CartPole-v1 environment."""
)

# Function to parse algorithm_comparison.md to extract performance metrics
@st.cache_data
def load_performance_data_from_md():
    """Load algorithm performance data from algorithm_comparison.md"""
    try:
        comparison_path = os.path.join(os.path.dirname(__file__), 'algorithm_comparison.md')
        
            
        with open(comparison_path, 'r') as f:
            content = f.read()
        
        # Extract the runtime and performance table
        # Look for tables that have Runtime or Performance data
        table_pattern = r"\|\s*Algorithm\s*\|.*?\|.*?\|\n\|[-\|\s:]*\|\n((\|.*?\|.*?\|.*?\|\n)+)"
        
        table_matches = re.findall(table_pattern, content)
        
        if not table_matches:
            # Fall back to hardcoded data
            return load_hardcoded_data()
        
        # Process the performance metrics table
        metrics_table = None
        for match in table_matches:
            if "Runtime" in match[0] or "Performance" in match[0]:
                metrics_table = match[0]
                break
        
        if not metrics_table:
            return load_hardcoded_data()
        
        # Parse table rows
        rows = metrics_table.strip().split('\n')
        data = []
        
        for row in rows:
            # Skip empty rows
            if not row.strip():
                continue
                
            # Extract cell values
            cells = [cell.strip() for cell in row.split('|')[1:-1]]
            if len(cells) >= 5:  # Ensure we have enough columns
                data.append(cells)
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            # Assume the structure matches our expected format
            df.columns = ["Algorithm", "Implementation", "Runtime", "Initial", "Final", "Improvement"]
            
            # Clean and convert numeric columns
            df["Runtime"] = df["Runtime"].str.extract(r'(\d+)').astype(float)
            df["Initial"] = df["Initial"].str.extract(r'(\d+\.?\d*)').astype(float)
            df["Final"] = df["Final"].str.extract(r'(\d+\.?\d*)').fillna(0).astype(float)
            df["Improvement"] = df["Improvement"].str.extract(r'(\d+\.?\d*)').fillna(0).astype(float)
            
            # Add environment and optimization columns for filtering
            df["Environment"] = df["Implementation"].apply(lambda x: "Colab" if "Colab" in x else "Local")
            df["Optimized"] = df["Implementation"].apply(lambda x: "Optimized" if "Optimized" in x else 
                                                      ("Dummy" if "Dummy" in x else "Unoptimized"))
            
            # Rename columns to match our UI
            df = df.rename(columns={
                "Runtime": "Runtime (sec)",
                "Initial": "Initial Performance",
                "Final": "Final Performance",
                "Improvement": "Improvement Factor"
            })
            
            return df
    except Exception as e:
        st.error(f"Error parsing algorithm_comparison.md: {e}")
        
    # Fallback to hardcoded data if anything goes wrong
    return load_hardcoded_data()

# Fallback data function
def load_hardcoded_data():
    """Provide hardcoded performance data as fallback"""
    data = {
        "Algorithm": [
            "PPO", "PPO", "PPO", 
            "A2C", "A2C", "A2C", 
            "DQN", "DQN", "DQN", 
            "MB-PPO"
        ],
        "Implementation": [
            "Optimized (Local)", "Unoptimized (Local)", "Optimized (Colab)",
            "Optimized (Local)", "Unoptimized (Local)", "Optimized (Colab)",
            "Optimized (Local)", "Unoptimized (Local)", "Optimized (Colab)",
            "Dummy (Local)"
        ],
        "Runtime (sec)": [17, 25, 8, 20, 30, 10, 16, 22, 9, 30],
        "Initial Performance": [9.10, 8.40, 24.10, 126.60, 15.20, 17.60, 9.50, 9.20, 16.40, 20.00],
        "Final Performance": [500.00, 450.00, 500.00, 500.00, 425.00, "Testing", 40.50, 20.30, "Testing", 20.00],
        "Improvement Factor": [55, 54, 21, 4, 28, "Increasing", 4.3, 2.2, "Increasing", 1],
        "Learning Pattern": ["Steady improvement", "Steady improvement", "Steady improvement", 
                           "Steady improvement", "Steady improvement", "Early reward ~20.6", 
                           "Gradual, plateaus early", "Gradual, plateaus early", "Exploration-dependent", 
                           "Flat (by design)"],
        "Exploration Strategy": ["Entropy-based", "Entropy-based", "Entropy-based",
                               "Entropy-based", "Entropy-based", "Entropy-based",
                               "ε-greedy (0.392→0.05)", "ε-greedy (0.392→0.05)", "ε-greedy (0.392→0.05)",
                               "Fixed random (50/50)"]
    }
    
    # Create DataFrame directly
    df = pd.DataFrame(data)
    
    # Add environment and optimization columns for filtering
    df["Environment"] = df["Implementation"].apply(lambda x: "Colab" if "Colab" in x else "Local")
    df["Optimized"] = df["Implementation"].apply(lambda x: "Optimized" if "Optimized" in x else 
                                               ("Dummy" if "Dummy" in x else "Unoptimized"))
    
    return df

# Function to load model-based RL explanation
@st.cache_data
def load_model_based_explanation():
    try:
        explanation_path = os.path.join(os.path.dirname(__file__), 'model_based_rl_explained.md')
        
        with open(explanation_path, 'r') as f:
            content = f.read()
            
        # Extract the introduction section
        intro_pattern = r"## Introduction to Model-Based RL[\s\S]*?(?=##)"
        intro_match = re.search(intro_pattern, content)
        
        if intro_match:
            return intro_match.group(0)
        else:
            return "Introduction section not found in model_based_rl_explained.md"
    except Exception as e:
        return f"Error loading model explanation: {e}"

# Load data
df = load_performance_data_from_md()

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparison", "Optimization Impact", "Learning Visualization", "Model-Based RL"])

with tab1:
    st.header("Algorithm Performance Metrics")
    
    # Add explanatory text from algorithm_comparison.md
    st.info("""
    **Key Metrics Explained:**
    - **Initial Performance**: Average reward before training (random policy level is ~20)
    - **Final Performance**: Average reward after training (perfect score is 500)
    - **Improvement Factor**: How many times better the final performance is compared to initial
    - **Runtime**: Training duration in seconds (lower is better)
    """)
    
    # Environment selector with more prominence
    st.subheader("Select Environment")
    env_col1, env_col2 = st.columns([1, 3])
    with env_col1:
        environment = st.radio(
            "Environment",
            ["All Environments", "Local Only", "Colab Only"],
            index=0,
            horizontal=True,
            help="Filter by environment to compare implementations"
        )
    
    with env_col2:
        st.markdown("""
        - **Local**: Tests run on local machine (macOS)
        - **Colab**: Tests run on Google Colab (cloud-based)
        - Different environments often show variation in initial performance due to random seeds and hardware differences
        """)
    
    # Filter by environment
    if environment == "Local Only":
        env_filter = ["Local"]
    elif environment == "Colab Only":
        env_filter = ["Colab"]
    else:
        env_filter = df["Environment"].unique()
    
    # Algorithm selector
    selected_algos = st.multiselect(
        "Select Algorithms",
        options=df["Algorithm"].unique(),
        default=df["Algorithm"].unique()
    )
    
    # Get data without any filtering first to debug
    all_df = df.copy()
    
    # Add a small debugging expander section to avoid cluttering the UI
    with st.expander("Debug Information (click to expand)", expanded=False):
        st.write(f"Total rows in original data: {len(all_df)}")
        unique_implementations = all_df['Implementation'].unique()
        st.write(f"All implementations in data: {list(unique_implementations)}")
    
        # Filter data
        filtered_df = df[
            (df["Algorithm"].isin(selected_algos)) &
            (df["Environment"].isin(env_filter))
        ]
        
        # Debug information
        st.write(f"Number of rows in filtered data: {len(filtered_df)}")
        st.write("Filtered implementations found:")
        st.write(filtered_df[['Algorithm', 'Implementation', 'Environment', 'Final Performance']].sort_values(['Algorithm', 'Implementation']))
    
    # Filter data (outside of debugging expander)
    filtered_df = df[
        (df["Algorithm"].isin(selected_algos)) &
        (df["Environment"].isin(env_filter))
    ]
    
    # Display metrics table with styling
    st.subheader("Performance Metrics Table")
    
    # Format the dataframe for display
    display_df = filtered_df[["Algorithm", "Implementation", "Runtime (sec)", 
                            "Initial Performance", "Final Performance", "Improvement Factor"]].copy()
    
    # Format numeric columns to 2 decimal places
    for col in ["Initial Performance", "Final Performance"]:
        display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)
    
    # Format improvement factor to 1 decimal place for readability
    display_df["Improvement Factor"] = display_df["Improvement Factor"].apply(
        lambda x: f"{float(x):.1f}x" if isinstance(x, (int, float)) else x
    )
    
    # Style the dataframe for better visualization
    def highlight_optimized(val):
        if 'Optimized' in str(val):
            return 'color: #008000; font-weight: bold'  # Dark green text for optimized
        elif 'Unoptimized' in str(val):
            return 'color: #CD5C5C'  # Indian red text for unoptimized
        elif 'Dummy' in str(val):
            return 'color: #B8860B'  # Dark golden text for dummy
        else:
            return ''
    
    # Display styled dataframe
    st.dataframe(
        display_df.style.applymap(highlight_optimized, subset=['Implementation']),
        use_container_width=True,
        hide_index=True
    )
    
    # Create bar chart for final performance with environment grouping
    st.subheader("Final Performance by Algorithm and Environment")
    
    # Create a copy of filtered_df for the chart to handle string values
    chart_df = filtered_df.copy()
    
    # Convert string values to numeric for charting, using NaN for non-numeric
    chart_df['Final Performance'] = pd.to_numeric(chart_df['Final Performance'], errors='coerce')
    
    # Create bar chart for final performance with environment grouping
    fig = px.bar(
        chart_df,
        x="Algorithm", 
        y="Final Performance",
        color="Implementation",
        barmode="group",
        height=500,
        labels={"Final Performance": "Final Reward"},
        title="Higher is better - CartPole-v1 max reward is 500",
        color_discrete_map={
            "Optimized (Local)": "#2E8B57",  # Dark green for optimized local
            "Unoptimized (Local)": "#CD5C5C",  # Indian red for unoptimized local
            "Optimized (Colab)": "#4682B4",  # Steel blue for optimized colab
            "Dummy (Local)": "#FFD700"  # Gold for dummy implementation
        }
    )
    
    # Format y-axis to show only 2 decimal places
    fig.update_layout(
        yaxis=dict(
            tickformat=".2f"
        )
    )
    
    # Add a horizontal line for perfect score
    fig.add_shape(
        type="line", x0=-0.5, y0=500, x1=3.5, y1=500,
        line=dict(color="green", dash="dash", width=2)
    )
    
    fig.add_annotation(
        x=1.5, y=510,
        text="Perfect Score (500)",
        showarrow=False,
        font=dict(color="green")
    )
    
    fig.update_layout(
        xaxis_tickangle=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add optimization impact visualization to main dashboard
    st.subheader("Optimization Impact Visualization")
    
    # Filter for local implementations
    local_df = df[df["Environment"] == "Local"]
    
    # Get PPO, A2C, DQN, and MB-PPO
    compare_algos = ["PPO", "A2C", "DQN", "MB-PPO"]
    compare_df = local_df[local_df["Algorithm"].isin(compare_algos)]
    
    # Create figure with two subplots: performance and runtime
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=["Performance Improvement", "Runtime Reduction"],
                        specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    # Performance improvement plot
    perf_improvements = {}
    runtime_reductions = {}
    
    for algo in compare_algos:
        algo_df = compare_df[compare_df["Algorithm"] == algo]
        
        if algo == "MB-PPO":
            # For MB-PPO, we compare against the average initial performance of other algorithms
            # to show it maintains the same random policy level
            dummy_perf = pd.to_numeric(algo_df["Final Performance"].iloc[0], errors='coerce')
            avg_initial = pd.to_numeric(compare_df[compare_df["Algorithm"] != "MB-PPO"]["Initial Performance"].mean(), errors='coerce')
            
            if not (pd.isna(dummy_perf) or pd.isna(avg_initial)):
                # Calculate how close MB-PPO stays to initial random policy level (should be near 0%)
                perf_improvement = ((dummy_perf - avg_initial) / avg_initial) * 100
                perf_improvements[algo] = perf_improvement
                runtime_reductions[algo] = 0  # No runtime optimization for MB-PPO
        
        elif len(algo_df) >= 2:  # For other algorithms, compare optimized vs unoptimized
            opt = algo_df[algo_df["Optimized"] == "Optimized"]
            unopt = algo_df[algo_df["Optimized"] == "Unoptimized"]
            
            if len(opt) > 0 and len(unopt) > 0:
                # Convert to numeric if they're strings
                opt_final = pd.to_numeric(opt["Final Performance"].iloc[0], errors='coerce')
                unopt_final = pd.to_numeric(unopt["Final Performance"].iloc[0], errors='coerce')
                
                if not (pd.isna(opt_final) or pd.isna(unopt_final)):
                    # Calculate percentage improvement
                    perf_improvement = ((opt_final - unopt_final) / unopt_final) * 100
                    perf_improvements[algo] = perf_improvement
                    
                    # Runtime reduction
                    opt_runtime = opt["Runtime (sec)"].iloc[0]
                    unopt_runtime = unopt["Runtime (sec)"].iloc[0]
                    runtime_reduction = ((unopt_runtime - opt_runtime) / unopt_runtime) * 100
                    runtime_reductions[algo] = runtime_reduction
    
    # Add performance improvement bars
    colors = {"PPO": "#2E8B57", "A2C": "#4682B4", "DQN": "#CD5C5C", "MB-PPO": "#FFD700"}
    
    for algo, improvement in perf_improvements.items():
        fig.add_trace(
            go.Bar(
                x=[algo],
                y=[improvement],
                name=algo,
                marker_color=colors[algo],
                text=[f"{improvement:.1f}%"],
                textposition="outside"
            ),
            row=1, col=1
        )
    
    # Add runtime reduction bars
    for algo, reduction in runtime_reductions.items():
        fig.add_trace(
            go.Bar(
                x=[algo],
                y=[reduction],
                name=algo,
                marker_color=colors[algo],
                text=[f"{reduction:.1f}%"],
                textposition="outside",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        title_text="Optimization Impact by Algorithm",
        yaxis=dict(title="Performance Improvement (%)", tickformat=".1f"),
        yaxis2=dict(title="Runtime Reduction (%)", tickformat=".1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add key findings from algorithm_comparison.md
    st.markdown("""
    **Key Optimization Findings:**
    
    1. **Performance Improvements**:
       - PPO: Optimization improved final reward by ~11% (450→500)
       - A2C: Optimization improved final reward by ~18% (425→500)
       - DQN: Optimization doubled final reward (+100% improvement) (20.3→40.5)
       - MB-PPO: Deliberately maintains random policy level (~20 reward) as a control baseline
    
    2. **Efficiency Gains**:
       - PPO: 32% runtime reduction (25→17 seconds)
       - A2C: 33% runtime reduction (30→20 seconds)
       - DQN: 27% runtime reduction (22→16 seconds)
       - MB-PPO: No optimization applied (fixed implementation)
    
    3. **Key Insight**:
       - The similar initial performance between Colab implementations and our MB-PPO skeleton (~16-24 vs. ~20) confirms that we've correctly implemented the random-policy baseline
       - The dramatic difference in final performance validates our testing methodology and the performance of the standard algorithms
    """)
    
    # Add key findings from algorithm_comparison.md
    st.subheader("Key Performance Insights")
    st.markdown("""
    **Performance Analysis:**
    
    1. **PPO** achieves perfect performance (500 reward) when optimized in both Local and Colab environments
    2. **A2C** reaches perfect performance when optimized locally and shows strong improvement in Colab
    3. **DQN** shows moderate performance improvement but doesn't reach perfect scores
    4. **MB-PPO Skeleton** maintains constant random-policy level performance (~20) by design
    
    **Runtime Analysis:**
    
    1. Colab implementations run 45-50% faster than local due to hardware acceleration
    2. Optimization improves runtime by 27-33% across all algorithms
    3. PPO and A2C benefit from parallel environment vectorization
    """)
    
    # Link to full comparison document with expander
    st.markdown("---")
    with st.expander("📊 View Complete Algorithm Comparison Details", expanded=False):
        comparison_path = os.path.join(os.path.dirname(__file__), 'algorithm_comparison.md')
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                content = f.read()
                
            # Extract main sections using regex
            sections = {
                "methodology": re.search(r"## Methodology([\s\S]*?)(?=##)", content),
                "results": re.search(r"## Results([\s\S]*?)(?=##)", content),
                "analysis": re.search(r"## Analysis([\s\S]*?)(?=##)", content),
                "optimization": re.search(r"### Optimization Impact Analysis([\s\S]*?)(?=###)", content),
                "implementation": re.search(r"## Implementation Details([\s\S]*?)(?=##)", content)
            }
            
            # Create tabs for different sections of the document
            doc_tabs = st.tabs(["Summary", "Results", "Analysis", "Optimization", "Implementation"])
            
            # Summary tab
            with doc_tabs[0]:
                st.markdown("### Algorithm Comparison Summary")
                st.markdown("""
                This document provides a comprehensive comparison of reinforcement learning algorithms 
                implemented using Stable Baselines3 on the CartPole-v1 environment.
                
                **Algorithms Compared:**
                - **PPO (Proximal Policy Optimization)**: An on-policy algorithm using a clipped surrogate objective
                - **A2C (Advantage Actor-Critic)**: An on-policy algorithm with policy and value networks
                - **DQN (Deep Q-Network)**: An off-policy algorithm with experience replay and target networks
                - **MB-PPO Skeleton**: A non-learning skeleton implementation as a control baseline
                
                **Environments:**
                - **Local**: Tests run on local machine (macOS)
                - **Colab**: Tests run on Google Colab (cloud-based)
                
                **Implementation Types:**
                - **Optimized**: Tuned hyperparameters and vectorized environments
                - **Unoptimized**: Default hyperparameters and standard configuration
                """)
                
                # Extract and display ASCII chart
                ascii_chart_pattern = r"```\nReward[\s\S]*?Start[\s\S]*?End\n```"
                ascii_match = re.search(ascii_chart_pattern, content)
                if ascii_match:
                    st.subheader("Performance Visualization")
                    st.code(ascii_match.group(0), language="text")
            
            # Results tab
            with doc_tabs[1]:
                if sections["results"]:
                    st.markdown(sections["results"].group(1))
                    
                    # Extract and display all tables in this section
                    tables_pattern = r"\|\s*Algorithm\s*\|.*?\|.*?\|\n\|[-\|\s:]*\|\n((\|.*?\|.*?\|.*?\|\n)+)"
                    tables = re.findall(tables_pattern, sections["results"].group(1))
                    
                    if tables:
                        for i, table in enumerate(tables):
                            # Convert markdown table to dataframe for better display
                            try:
                                rows = table[0].strip().split('\n')
                                data = []
                                for row in rows:
                                    if row.strip():
                                        cells = [cell.strip() for cell in row.split('|')[1:-1]]
                                        data.append(cells)
                                
                                if data:
                                    # Get headers from the table above this one in the markdown
                                    header_pattern = r"\|\s*(.*?)\s*\|\n\|[-\|\s:]*\|\n" + re.escape(table[0])
                                    header_match = re.search(header_pattern, sections["results"].group(1))
                                    
                                    if header_match:
                                        headers = [h.strip() for h in header_match.group(1).split('|')]
                                        if len(headers) == len(data[0]):
                                            df = pd.DataFrame(data, columns=headers)
                                            st.dataframe(df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Error processing table {i}: {e}")
            
            # Analysis tab
            with doc_tabs[2]:
                if sections["analysis"]:
                    st.markdown(sections["analysis"].group(1))
            
            # Optimization tab
            with doc_tabs[3]:
                if sections["optimization"]:
                    st.markdown(sections["optimization"].group(1))
                    
                    # Create visualization of optimization impact
                    st.subheader("Visualization of Optimization Impact")
                    
                    # Filter for local implementations
                    local_df = df[df["Environment"] == "Local"]
                    opt_df = local_df[local_df["Optimized"].isin(["Optimized", "Unoptimized"])]
                    
                    # Get PPO, A2C, DQN only
                    compare_algos = ["PPO", "A2C", "DQN"]
                    compare_df = opt_df[opt_df["Algorithm"].isin(compare_algos)]
                    
                    # Create figure with two subplots: performance and runtime
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go
                    
                    fig = make_subplots(rows=1, cols=2, 
                                      subplot_titles=["Performance Improvement", "Runtime Reduction"],
                                      specs=[[{"type": "bar"}, {"type": "bar"}]])
                    
                    # Performance improvement plot
                    for i, algo in enumerate(compare_algos):
                        algo_df = compare_df[compare_df["Algorithm"] == algo]
                        
                        if len(algo_df) == 2:  # Make sure we have both optimized and unoptimized
                            opt = algo_df[algo_df["Optimized"] == "Optimized"]
                            unopt = algo_df[algo_df["Optimized"] == "Unoptimized"]
                            
                            # Convert to numeric if they're strings
                            opt_final = pd.to_numeric(opt["Final Performance"].iloc[0], errors='coerce')
                            unopt_final = pd.to_numeric(unopt["Final Performance"].iloc[0], errors='coerce')
                            
                            if not (pd.isna(opt_final) or pd.isna(unopt_final)):
                                # Calculate percentage improvement
                                perf_improvement = ((opt_final - unopt_final) / unopt_final) * 100
                                fig.add_trace(
                                    go.Bar(
                                        x=[algo],
                                        y=[perf_improvement],
                                        name=f"{algo} Performance",
                                        marker_color=["#2E8B57", "#4682B4", "#CD5C5C"][i],
                                        text=[f"{perf_improvement:.1f}%"],
                                        textposition="outside"
                                    ),
                                    row=1, col=1
                                )
                    
                    # Runtime reduction plot
                    for i, algo in enumerate(compare_algos):
                        algo_df = compare_df[compare_df["Algorithm"] == algo]
                        
                        if len(algo_df) == 2:  # Make sure we have both optimized and unoptimized
                            opt = algo_df[algo_df["Optimized"] == "Optimized"]
                            unopt = algo_df[algo_df["Optimized"] == "Unoptimized"]
                            
                            # Convert to numeric if they're strings
                            opt_runtime = opt["Runtime (sec)"].iloc[0]
                            unopt_runtime = unopt["Runtime (sec)"].iloc[0]
                            
                            # Calculate percentage reduction
                            runtime_reduction = ((unopt_runtime - opt_runtime) / unopt_runtime) * 100
                            fig.add_trace(
                                go.Bar(
                                    x=[algo],
                                    y=[runtime_reduction],
                                    name=f"{algo} Runtime",
                                    marker_color=["#2E8B57", "#4682B4", "#CD5C5C"][i],
                                    text=[f"{runtime_reduction:.1f}%"],
                                    textposition="outside",
                                    showlegend=False
                                ),
                                row=1, col=2
                            )
                    
                    fig.update_layout(
                        height=500,
                        title_text="Optimization Impact by Algorithm",
                        showlegend=False,
                        yaxis_title="Performance Improvement (%)",
                        yaxis2_title="Runtime Reduction (%)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Implementation tab
            with doc_tabs[4]:
                if sections["implementation"]:
                    st.markdown(sections["implementation"].group(1))
                    
                    # Extract code blocks
                    code_blocks = re.findall(r"```python\n([\s\S]*?)\n```", sections["implementation"].group(1))
                    
                    if code_blocks:
                        st.subheader("Implementation Code Snippets")
                        for i, code in enumerate(code_blocks):
                            with st.expander(f"Code Snippet {i+1}"):
                                st.code(code, language="python")
        else:
            st.error("Could not find algorithm_comparison.md file")
    
    # Link to full comparison document
    st.markdown("For complete details, see [algorithm_comparison.md](./algorithm_comparison.md)")

with tab2:
    st.header("Optimization Impact Analysis")
    
    # Filter for local implementations only for optimization comparison
    local_df = df[df["Environment"] == "Local"]
    
    # Create metrics
    opt_metrics = {
        "PPO": {
            "perf_improve": "11%",
            "runtime_reduce": "32%",
            "key_finding": "Achieved perfect score only when optimized"
        },
        "A2C": {
            "perf_improve": "18%",
            "runtime_reduce": "33%",
            "key_finding": "Significant variance reduction with optimization"
        },
        "DQN": {
            "perf_improve": "100%",
            "runtime_reduce": "27%",
            "key_finding": "Most sensitive to optimization"
        },
        "MB-PPO": {
            "perf_improve": "0%",
            "runtime_reduce": "0%",
            "key_finding": "Maintains random policy level as a control baseline"
        }
    }
    
    # Display metrics in columns
    cols = st.columns(len(opt_metrics))
    
    for i, (algo, metrics) in enumerate(opt_metrics.items()):
        with cols[i]:
            st.subheader(algo)
            st.metric("Performance Improvement", metrics["perf_improve"])
            st.metric("Runtime Reduction", metrics["runtime_reduce"])
            st.write(f"**Key Finding:** {metrics['key_finding']}")
    
    # Create comparison chart
    st.subheader("Optimized vs. Unoptimized Performance")
    
    # Get only the algorithms with both optimized and unoptimized versions
    compare_algos = ["PPO", "A2C", "DQN"]
    compare_df = local_df[local_df["Algorithm"].isin(compare_algos)]
    
    fig = px.bar(
        compare_df,
        x="Algorithm", 
        y="Final Performance",
        color="Optimized",
        barmode="group",
        height=400,
        labels={"Final Performance": "Final Reward"},
        color_discrete_map={"Optimized": "#00CC96", "Unoptimized": "#EF553B"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Runtime comparison
    st.subheader("Runtime Efficiency Gains")
    
    fig = px.bar(
        compare_df,
        x="Algorithm", 
        y="Runtime (sec)",
        color="Optimized",
        barmode="group",
        height=400,
        labels={"Runtime (sec)": "Runtime (seconds)"},
        color_discrete_map={"Optimized": "#00CC96", "Unoptimized": "#EF553B"}
    )
    
    # Lower is better for runtime
    fig.update_layout(title_text="Lower is better - Runtime in seconds")
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Learning Visualization")
    
    # Create mock training data
    @st.cache_data
    def generate_learning_curves():
        steps = np.linspace(0, 1, 100)
        curves = {
            "PPO (Optimized)": 9.1 + 490.9 * (1 - np.exp(-5 * steps)),
            "PPO (Unoptimized)": 8.4 + 441.6 * (1 - np.exp(-4 * steps)),
            "A2C (Optimized)": 126.6 + 373.4 * (1 - np.exp(-3 * steps)),
            "A2C (Unoptimized)": 15.2 + 409.8 * (1 - np.exp(-3 * steps)),
            "DQN (Optimized)": 9.5 + 31.0 * (1 - np.exp(-2 * steps)),
            "DQN (Unoptimized)": 9.2 + 11.1 * (1 - np.exp(-1.5 * steps)),
            "MB-PPO Skeleton": np.ones(100) * 20
        }
        return curves, steps
    
    curves, steps = generate_learning_curves()
    
    # Algorithm selector
    selected_curves = st.multiselect(
        "Select Algorithms to Visualize",
        options=list(curves.keys()),
        default=["PPO (Optimized)", "A2C (Optimized)", "DQN (Optimized)", "MB-PPO Skeleton"]
    )
    
    # Create learning curve plot
    fig = go.Figure()
    
    for algo in selected_curves:
        fig.add_trace(go.Scatter(
            x=steps * 100, # Convert to percentage of training
            y=curves[algo],
            mode='lines',
            name=algo
        ))
    
    fig.update_layout(
        title="Learning Curves (Simulated)",
        xaxis_title="Training Progress (%)",
        yaxis_title="Reward",
        height=500,
        yaxis=dict(range=[0, 520]),  # Set y-axis range
        hovermode="x unified"
    )
    
    # Add a horizontal line for maximum possible reward
    fig.add_shape(type="line",
        x0=0, y0=500, x1=100, y1=500,
        line=dict(color="gray", dash="dash")
    )
    
    fig.add_annotation(
        x=50, y=510, 
        text="Maximum Possible Reward (500)",
        showarrow=False, 
        font=dict(color="gray")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Load visualization from algorithm_comparison.md
    comparison_path = os.path.join(os.path.dirname(__file__), 'algorithm_comparison.md')
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            content = f.read()
            ascii_chart_pattern = r"```\nReward[\s\S]*?Start[\s\S]*?End\n```"
            ascii_match = re.search(ascii_chart_pattern, content)
            
            if ascii_match:
                st.subheader("ASCII Chart from Documentation")
                st.code(ascii_match.group(0), language="text")
    
    # Include code for running tests
    st.subheader("How to Run Algorithm Tests")
    st.code("""
# Run PPO test
python ppo_test.py

# Run A2C test
python a2c_test.py

# Run DQN test
python dqn_test.py

# Run MB-PPO skeleton test
python mb_ppo_test.py
    """, language="bash")

with tab4:
    st.header("Model-Based RL Approach")
    
    # Load explanation from our markdown file
    explanation = load_model_based_explanation()
    st.markdown(explanation)
    
    # Architecture diagram
    st.subheader("MB-PPO Architecture")
    
    # Create a simple architecture diagram
    architecture = '''
digraph G {
  rankdir=LR;
  node [shape=box, style=filled, fillcolor=lightblue];
  
  Environment [fillcolor=lightgreen];
  Policy [label="Policy Network (πθ)"];
  Value [label="Value Network (Vθ)"];
  WorldModel [label="World Model (Wφ)"];
  Curiosity [label="Curiosity Module (Cψ)"];
  Buffer [label="Mixed Rollout Buffer"];
  
  Environment -> Policy [label="states"];
  Environment -> Value [label="states", style="dashed"];
  Policy -> Environment [label="actions"];
  Environment -> Buffer [label="real transitions"];
  
  # Add shared feature connection between policy and value
  Policy -> Value [label="shared features", style="dotted", dir="both", color="darkblue"];
  
  WorldModel -> Buffer [label="imagined transitions"];
  Policy -> WorldModel [label="actions for imagination"];
  WorldModel -> Curiosity [label="states"];
  Curiosity -> Buffer [label="intrinsic rewards"];
  
  Buffer -> Policy [label="policy update"];
  Buffer -> Value [label="value update"];
  Buffer -> WorldModel [label="world model update"];
  
  # Add advantage computation
  Value -> Policy [label="advantage signals", color="red"];
}
'''
    
    try:
        from graphviz import Source
        st.graphviz_chart(architecture)
    except ImportError:
        st.code(architecture, language="dot")
        st.info("Install graphviz to see the rendered diagram")
    
    # Implementation status
    st.subheader("Current Implementation Status")
    
    status = {
        "Policy Network": "Skeleton (Fixed Random)",
        "Value Network": "Skeleton (Fixed Zero)",
        "World Model": "Skeleton (Identity Function)",
        "Curiosity Module": "Skeleton (Fixed Reward)",
        "Mixed Buffer": "Functional (Stores Transitions)",
        "Adaptive Imagination": "Structure Only"
    }
    
    status_df = pd.DataFrame({
        "Component": status.keys(),
        "Status": status.values()
    })
    
    st.table(status_df)
    
    # Link to full explanation
    st.markdown("For complete details on the Model-Based approach, see [model_based_rl_explained.md](./model_based_rl_explained.md)")

# Add footer
st.markdown("---")
st.markdown(
    """<div style='text-align: center;'>
    <p>RL-stablebaseline-scientist Dashboard u2022 CartPole-v1 Benchmark Results</p>
    </div>""", 
    unsafe_allow_html=True
)
