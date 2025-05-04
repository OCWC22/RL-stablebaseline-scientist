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
            st.warning("Could not find performance tables in algorithm_comparison.md")
            # Fall back to hardcoded data
            return load_hardcoded_data()
        
        # Process the performance metrics table
        metrics_table = None
        for match in table_matches:
            if "Runtime" in match[0] or "Performance" in match[0]:
                metrics_table = match[0]
                break
        
        if not metrics_table:
            st.warning("Could not find performance metrics table in algorithm_comparison.md")
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
        "Final Performance": [500.00, 450.00, 500.00, 500.00, 425.00, 0.00, 40.50, 20.30, 0.00, 20.00],
        "Improvement Factor": [55, 54, 21, 4, 28, 0, 4.3, 2.2, 0, 1],
    }
    
    # Replace 0 values for ongoing tests with NaN
    df = pd.DataFrame(data)
    df["Final Performance"] = df["Final Performance"].replace(0, np.nan)
    df["Improvement Factor"] = df["Improvement Factor"].replace(0, np.nan)
    
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
        
        if not os.path.exists(explanation_path):
            return "Could not find model_based_rl_explained.md"
            
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
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_algos = st.multiselect(
            "Select Algorithms",
            options=df["Algorithm"].unique(),
            default=df["Algorithm"].unique()
        )
    
    with col2:
        selected_env = st.multiselect(
            "Select Environments",
            options=df["Environment"].unique(),
            default=df["Environment"].unique()
        )
    
    # Filter data
    filtered_df = df[
        (df["Algorithm"].isin(selected_algos)) &
        (df["Environment"].isin(selected_env))
    ]
    
    # Display metrics table
    st.dataframe(
        filtered_df[["Algorithm", "Implementation", "Runtime (sec)", 
                 "Initial Performance", "Final Performance", "Improvement Factor"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Create bar chart for final performance
    st.subheader("Final Performance by Algorithm")
    
    fig = px.bar(
        filtered_df,
        x="Implementation", 
        y="Final Performance",
        color="Algorithm",
        barmode="group",
        height=400,
        labels={"Final Performance": "Final Reward", "Implementation": ""},
        title="Higher is better - CartPole-v1 max reward is 500"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
  Policy -> Environment [label="actions"];
  Environment -> Buffer [label="real transitions"];
  
  WorldModel -> Buffer [label="imagined transitions"];
  Policy -> WorldModel [label="actions for imagination"];
  WorldModel -> Curiosity [label="states"];
  Curiosity -> Buffer [label="intrinsic rewards"];
  
  Buffer -> Policy [label="policy update"];
  Buffer -> Value [label="value update"];
  Buffer -> WorldModel [label="world model update"];
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
