import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import json
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Safe import for serial
try:
    import serial
except ImportError:
    serial = None

st.set_page_config(page_title="Battery Management System", layout="wide")
st.title("🔋 Battery Management System - Enhanced Dashboard")

# Initialize session state
if 'task_running' not in st.session_state:
    st.session_state.task_running = False
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = "Idle"
if 'phase_start_time' not in st.session_state:
    st.session_state.phase_start_time = None
if 'task_config' not in st.session_state:
    st.session_state.task_config = {}
if 'cell_configs' not in st.session_state:
    st.session_state.cell_configs = ["LFP"] * 8
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []

# Sidebar navigation
page = st.sidebar.selectbox("Choose Dashboard", [
    "📊 Real-Time Monitoring",
    "🔧 Cell Configuration", 
    "⚡ Task Controller",
    "📈 Data Analysis",
    "📝 Data Management"
])

# Function to generate cell data based on phase and cell type
def generate_cell_data(cell_type, phase, cell_id):
    base_voltage = 3.2 if cell_type == "LFP" else 3.6
    
    if phase == "Charging":
        voltage = base_voltage + random.uniform(0.1, 0.4)
        current = random.uniform(1.0, 3.0)  # Positive for charging
        temp_increase = random.uniform(2, 8)
    elif phase == "Discharging":
        voltage = base_voltage - random.uniform(0.0, 0.2)
        current = -random.uniform(1.0, 3.0)  # Negative for discharging
        temp_increase = random.uniform(1, 5)
    else:  # Idle
        voltage = base_voltage + random.uniform(-0.1, 0.1)
        current = random.uniform(-0.1, 0.1)  # Nearly zero
        temp_increase = 0
    
    return {
        "Cell_ID": f"Cell_{cell_id+1}",
        "Cell_Type": cell_type,
        "Voltage": round(voltage, 3),
        "Current": round(current, 2),
        "Temperature": round(25.0 + temp_increase + random.uniform(-2, 2), 1),
        "Power": round(voltage * abs(current), 2),
        "SOC": round(random.uniform(20, 95), 1),
        "SOH": round(random.uniform(85, 100), 1),
        "Internal_Resistance": round(random.uniform(10, 30), 1),
        "Phase": phase,
        "Timestamp": datetime.datetime.now().isoformat()
    }

# Function to save data to CSV
def save_to_csv(data, filename="battery_data_log.csv"):
    df = pd.DataFrame(data)
    
    # Check if file exists
    if os.path.exists(filename):
        # Append to existing file
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        df.to_csv(filename, mode='w', header=True, index=False)
    
    return filename

# Task execution logic
def execute_task():
    config = st.session_state.task_config
    current_time = time.time()
    
    if not st.session_state.task_running:
        return "Task Not Started"
    
    if st.session_state.phase_start_time is None:
        st.session_state.phase_start_time = current_time
        st.session_state.current_phase = "Charging"
        return "Charging"
    
    elapsed = current_time - st.session_state.phase_start_time
    
    if st.session_state.current_phase == "Charging" and elapsed >= config.get('charge_time', 10):
        st.session_state.current_phase = "Idle"
        st.session_state.phase_start_time = current_time
    elif st.session_state.current_phase == "Idle" and elapsed >= config.get('idle_time', 5):
        st.session_state.current_phase = "Discharging"
        st.session_state.phase_start_time = current_time
    elif st.session_state.current_phase == "Discharging" and elapsed >= config.get('discharge_time', 10):
        st.session_state.task_running = False
        st.session_state.current_phase = "Task Complete"
        st.session_state.phase_start_time = None
        return "Task Complete"
    
    return st.session_state.current_phase

# ================================ REAL-TIME MONITORING PAGE ================================
if page == "📊 Real-Time Monitoring":
    st.header("Real-Time Battery Monitoring")
    
    # Task status display
    current_phase = execute_task()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Phase", current_phase)
    with col2:
        if st.session_state.task_running and st.session_state.phase_start_time:
            elapsed = time.time() - st.session_state.phase_start_time
            st.metric("Phase Duration (s)", f"{elapsed:.1f}")
        else:
            st.metric("Phase Duration (s)", "0.0")
    with col3:
        st.metric("Task Status", "Running" if st.session_state.task_running else "Stopped")
    
    # Generate current cell data
    current_data = []
    for i in range(8):
        cell_data = generate_cell_data(st.session_state.cell_configs[i], current_phase, i)
        current_data.append(cell_data)
    
    # Add to historical data
    st.session_state.historical_data.extend(current_data)
    
    # Keep only last 1000 records to prevent memory issues
    if len(st.session_state.historical_data) > 1000:
        st.session_state.historical_data = st.session_state.historical_data[-1000:]
    
    # Display current data
    df_current = pd.DataFrame(current_data)
    st.subheader("Current Cell Status")
    st.dataframe(df_current, use_container_width=True)
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Voltage by Cell")
        fig = px.bar(df_current, x='Cell_ID', y='Voltage', color='Cell_Type',
                     title="Current Cell Voltages")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Current Flow")
        fig = px.bar(df_current, x='Cell_ID', y='Current', color='Phase',
                     title="Current Flow by Cell")
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature and Power charts
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Temperature Distribution")
        fig = px.scatter(df_current, x='Cell_ID', y='Temperature', color='Cell_Type',
                        size='Power', title="Cell Temperatures")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("Power Output")
        fig = px.line(df_current, x='Cell_ID', y='Power', color='Phase',
                      title="Power by Cell", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh
    if st.session_state.task_running:
        time.sleep(1)
        st.rerun()

# ================================ CELL CONFIGURATION PAGE ================================
elif page == "🔧 Cell Configuration":
    st.header("Cell Configuration")
    
    st.subheader("Configure Individual Cells")
    cols = st.columns(4)
    
    for i in range(8):
        with cols[i % 4]:
            cell_type = st.selectbox(
                f"Cell {i+1} Type", 
                ["LFP", "NMC", "LTO"], 
                index=0 if st.session_state.cell_configs[i] == "LFP" else 1,
                key=f"cell_config_{i}"
            )
            st.session_state.cell_configs[i] = cell_type
    
    # Display configuration summary
    st.subheader("Configuration Summary")
    config_data = []
    for i, cell_type in enumerate(st.session_state.cell_configs):
        config_data.append({
            "Cell": f"Cell_{i+1}",
            "Type": cell_type,
            "Nominal_Voltage": "3.2V" if cell_type == "LFP" else "3.6V" if cell_type == "NMC" else "2.4V",
            "Chemistry": "Lithium Iron Phosphate" if cell_type == "LFP" else "Lithium Nickel Manganese Cobalt Oxide" if cell_type == "NMC" else "Lithium Titanate"
        })
    
    df_config = pd.DataFrame(config_data)
    st.dataframe(df_config, use_container_width=True)
    
    # Configuration actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset to All LFP"):
            st.session_state.cell_configs = ["LFP"] * 8
            st.rerun()
    
    with col2:
        if st.button("Set Mixed Configuration"):
            st.session_state.cell_configs = ["LFP", "NMC", "LFP", "NMC", "LFP", "NMC", "LFP", "NMC"]
            st.rerun()
    
    with col3:
        if st.button("Random Configuration"):
            st.session_state.cell_configs = [random.choice(["LFP", "NMC", "LTO"]) for _ in range(8)]
            st.rerun()

# ================================ TASK CONTROLLER PAGE ================================
elif page == "⚡ Task Controller":
    st.header("Battery Task Controller")
    
    # Task configuration
    st.subheader("Configure Task Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        charge_time = st.number_input("Charging Duration (seconds)", min_value=1, max_value=300, value=10)
    
    with col2:
        idle_time = st.number_input("Idle Duration (seconds)", min_value=1, max_value=60, value=5)
    
    with col3:
        discharge_time = st.number_input("Discharge Duration (seconds)", min_value=1, max_value=300, value=10)
    
    # Update task configuration
    st.session_state.task_config = {
        'charge_time': charge_time,
        'idle_time': idle_time,
        'discharge_time': discharge_time
    }
    
    # Task controls
    st.subheader("Task Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Start Task", type="primary"):
            st.session_state.task_running = True
            st.session_state.phase_start_time = None
            st.session_state.current_phase = "Starting"
            st.success("Task started!")
            st.rerun()
    
    with col2:
        if st.button("Stop Task", type="secondary"):
            st.session_state.task_running = False
            st.session_state.phase_start_time = None
            st.session_state.current_phase = "Stopped"
            st.warning("Task stopped!")
    
    with col3:
        if st.button("Reset Task"):
            st.session_state.task_running = False
            st.session_state.phase_start_time = None
            st.session_state.current_phase = "Idle"
            st.info("Task reset!")
    
    with col4:
        if st.button("Emergency Stop", type="secondary"):
            st.session_state.task_running = False
            st.session_state.phase_start_time = None
            st.session_state.current_phase = "Emergency Stop"
            st.error("Emergency stop activated!")
    
    # Task progress visualization
    st.subheader("Task Progress")
    total_time = charge_time + idle_time + discharge_time
    
    progress_data = {
        "Phase": ["Charging", "Idle", "Discharging"],
        "Duration": [charge_time, idle_time, discharge_time],
        "Percentage": [charge_time/total_time*100, idle_time/total_time*100, discharge_time/total_time*100]
    }
    
    fig = px.pie(pd.DataFrame(progress_data), values='Duration', names='Phase', 
                 title="Task Phase Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Manual controls
    st.subheader("Manual Cell Controls")
    selected_cells = st.multiselect("Select Cells to Control", 
                                   [f"Cell_{i+1}" for i in range(8)], 
                                   default=[])
    
    if selected_cells:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Manual Charge Selected"):
                st.info(f"Charging {', '.join(selected_cells)}")
        with col2:
            if st.button("Manual Discharge Selected"):
                st.info(f"Discharging {', '.join(selected_cells)}")
        with col3:
            if st.button("Set Idle Selected"):
                st.info(f"Setting {', '.join(selected_cells)} to idle")

# ================================ DATA ANALYSIS PAGE ================================
elif page == "📈 Data Analysis":
    st.header("Data Analysis Dashboard")
    
    if not st.session_state.historical_data:
        st.warning("No historical data available. Run some tasks first!")
    else:
        df_hist = pd.DataFrame(st.session_state.historical_data)
        df_hist['Timestamp'] = pd.to_datetime(df_hist['Timestamp'])
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df_hist))
        with col2:
            st.metric("Unique Phases", df_hist['Phase'].nunique())
        with col3:
            st.metric("Time Span", f"{(df_hist['Timestamp'].max() - df_hist['Timestamp'].min()).total_seconds():.0f}s")
        with col4:
            st.metric("Avg Power", f"{df_hist['Power'].mean():.2f}W")
        
        # Interactive filters
        st.subheader("Data Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_cells = st.multiselect("Select Cells", df_hist['Cell_ID'].unique(), 
                                          default=df_hist['Cell_ID'].unique())
        with col2:
            selected_phases = st.multiselect("Select Phases", df_hist['Phase'].unique(),
                                           default=df_hist['Phase'].unique())
        with col3:
            selected_types = st.multiselect("Select Cell Types", df_hist['Cell_Type'].unique(),
                                          default=df_hist['Cell_Type'].unique())
        
        # Filter data
        filtered_df = df_hist[
            (df_hist['Cell_ID'].isin(selected_cells)) &
            (df_hist['Phase'].isin(selected_phases)) &
            (df_hist['Cell_Type'].isin(selected_types))
        ]
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters!")
        else:
            # Multiple chart types
            chart_type = st.selectbox("Select Chart Type", 
                                    ["Line Chart", "Scatter Plot", "Box Plot", "Heatmap", "3D Scatter"])
            
            if chart_type == "Line Chart":
                parameter = st.selectbox("Select Parameter", 
                                       ["Voltage", "Current", "Temperature", "Power", "SOC"])
                fig = px.line(filtered_df, x='Timestamp', y=parameter, color='Cell_ID',
                             facet_col='Phase', title=f"{parameter} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Scatter Plot":
                x_param = st.selectbox("X-axis", ["Voltage", "Current", "Temperature", "Power"], index=0)
                y_param = st.selectbox("Y-axis", ["Voltage", "Current", "Temperature", "Power"], index=3)
                fig = px.scatter(filtered_df, x=x_param, y=y_param, color='Phase', 
                               size='SOC', hover_data=['Cell_ID'], title=f"{x_param} vs {y_param}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Box Plot":
                parameter = st.selectbox("Select Parameter", 
                                       ["Voltage", "Current", "Temperature", "Power"])
                fig = px.box(filtered_df, x='Phase', y=parameter, color='Cell_Type',
                           title=f"{parameter} Distribution by Phase")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Heatmap":
                # Create correlation heatmap
                numeric_cols = ['Voltage', 'Current', 'Temperature', 'Power', 'SOC', 'SOH']
                corr_matrix = filtered_df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Parameter Correlation Heatmap",
                               color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "3D Scatter":
                fig = px.scatter_3d(filtered_df, x='Voltage', y='Current', z='Temperature',
                                  color='Phase', size='Power', hover_data=['Cell_ID'],
                                  title="3D Cell Parameter Visualization")
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            st.subheader("Statistical Analysis")
            st.dataframe(filtered_df.describe(), use_container_width=True)

# ================================ DATA MANAGEMENT PAGE ================================
elif page == "📝 Data Management":
    st.header("Data Management")
    
    # Current session data
    st.subheader("Current Session Data")
    if st.session_state.historical_data:
        df_session = pd.DataFrame(st.session_state.historical_data)
        st.dataframe(df_session.tail(50), use_container_width=True)  # Show last 50 records
        
        # Data export options
        st.subheader("Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save to CSV"):
                filename = save_to_csv(st.session_state.historical_data)
                st.success(f"Data saved to {filename}")
        
        with col2:
            csv_data = df_session.to_csv(index=False)
            st.download_button("Download CSV", csv_data, "battery_data.csv", "text/csv")
        
        with col3:
            json_data = df_session.to_json(orient='records', indent=2)
            st.download_button("Download JSON", json_data, "battery_data.json", "application/json")
        
        # Data management actions
        st.subheader("Data Management Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Session Data"):
                st.session_state.historical_data = []
                st.success("Session data cleared!")
                st.rerun()
        
        with col2:
            data_limit = st.number_input("Set Data Limit", min_value=100, max_value=5000, value=1000)
            if st.button("Apply Limit"):
                if len(st.session_state.historical_data) > data_limit:
                    st.session_state.historical_data = st.session_state.historical_data[-data_limit:]
                    st.info(f"Data limited to last {data_limit} records")
        
        with col3:
            if st.button("Generate Sample Data"):
                sample_data = []
                for _ in range(100):
                    for i in range(8):
                        sample_data.append(generate_cell_data(
                            st.session_state.cell_configs[i], 
                            random.choice(["Charging", "Discharging", "Idle"]), 
                            i
                        ))
                st.session_state.historical_data.extend(sample_data)
                st.success("Sample data generated!")
                st.rerun()
    
    else:
        st.info("No data available in current session. Start monitoring to collect data.")
    
    # File management
    st.subheader("File Management")
    log_file = "battery_data_log.csv"
    
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        st.info(f"Log file exists: {log_file} ({file_size} bytes)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Log File"):
                try:
                    df_log = pd.read_csv(log_file)
                    st.dataframe(df_log.tail(100), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading log file: {e}")
        
        with col2:
            if st.button("Delete Log File"):
                try:
                    os.remove(log_file)
                    st.success("Log file deleted!")
                except Exception as e:
                    st.error(f"Error deleting file: {e}")
    else:
        st.info("No log file exists yet.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Enhanced BMS v2.0 - More Interactive & Feature Rich")
