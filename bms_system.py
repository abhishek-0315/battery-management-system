import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import json
import time
import os

# Safe imports with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not available. Install with: pip install plotly")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Battery Management System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔋 Battery Management System - Enhanced Dashboard")

# Initialize session state with proper error handling
def init_session_state():
    defaults = {
        'task_running': False,
        'current_phase': "Idle",
        'phase_start_time': None,
        'task_config': {'charge_time': 10, 'idle_time': 5, 'discharge_time': 10},
        'cell_configs': ["LFP"] * 8,
        'historical_data': [],
        'last_update': time.time()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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
    try:
        base_voltage = 3.2 if cell_type == "LFP" else 3.6 if cell_type == "NMC" else 2.4
        
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
        
        power = voltage * abs(current)
        
        return {
            "Cell_ID": f"Cell_{cell_id+1}",
            "Cell_Type": cell_type,
            "Voltage": round(voltage, 3),
            "Current": round(current, 2),
            "Temperature": round(25.0 + temp_increase + random.uniform(-2, 2), 1),
            "Power": round(power, 2),
            "SOC": round(random.uniform(20, 95), 1),
            "SOH": round(random.uniform(85, 100), 1),
            "Internal_Resistance": round(random.uniform(10, 30), 1),
            "Phase": phase,
            "Timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error generating cell data: {e}")
        return None

# Function to save data to CSV with error handling
def save_to_csv(data, filename="battery_data_log.csv"):
    try:
        if not data:
            st.warning("No data to save")
            return None
            
        df = pd.DataFrame(data)
        
        # Check if file exists
        if os.path.exists(filename):
            # Append to existing file
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            # Create new file with headers
            df.to_csv(filename, mode='w', header=True, index=False)
        
        return filename
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")
        return None

# Task execution logic with improved error handling
def execute_task():
    try:
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
    except Exception as e:
        st.error(f"Error in task execution: {e}")
        return "Error"

# Create fallback charts if Plotly is not available
def create_fallback_chart(df, chart_type, x_col, y_col, title):
    if PLOTLY_AVAILABLE:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        if chart_type in ["bar", "line"]:
            st.bar_chart(df.set_index(x_col)[y_col])
        else:
            st.line_chart(df.set_index(x_col)[y_col])

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
        if cell_data:
            current_data.append(cell_data)
    
    if current_data:
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
            create_fallback_chart(df_current, "bar", "Cell_ID", "Voltage", "Current Cell Voltages")
        
        with col2:
            create_fallback_chart(df_current, "bar", "Cell_ID", "Current", "Current Flow by Cell")
        
        # Temperature and Power charts
        col3, col4 = st.columns(2)
        
        with col3:
            create_fallback_chart(df_current, "scatter", "Cell_ID", "Temperature", "Cell Temperatures")
        
        with col4:
            create_fallback_chart(df_current, "line", "Cell_ID", "Power", "Power by Cell")
        
        # Pack summary
        st.subheader("Pack Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_voltage = df_current["Voltage"].sum()
            st.metric("Total Voltage (V)", f"{total_voltage:.2f}")
        with col2:
            total_current = df_current["Current"].sum()
            st.metric("Total Current (A)", f"{total_current:.2f}")
        with col3:
            total_power = df_current["Power"].sum()
            st.metric("Total Power (W)", f"{total_power:.2f}")
        with col4:
            avg_temp = df_current["Temperature"].mean()
            st.metric("Avg Temperature (°C)", f"{avg_temp:.1f}")
    
    # Auto-refresh control
    if st.session_state.task_running:
        # Update every 2 seconds to prevent too frequent refreshes
        if time.time() - st.session_state.last_update > 2:
            st.session_state.last_update = time.time()
            st.rerun()

# ================================ CELL CONFIGURATION PAGE ================================
elif page == "🔧 Cell Configuration":
    st.header("Cell Configuration")
    
    st.subheader("Configure Individual Cells")
    
    # Create configuration in a more stable way
    for i in range(8):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**Cell {i+1}:**")
        with col2:
            current_index = 0
            if st.session_state.cell_configs[i] == "NMC":
                current_index = 1
            elif st.session_state.cell_configs[i] == "LTO":
                current_index = 2
                
            cell_type = st.selectbox(
                f"Type", 
                ["LFP", "NMC", "LTO"], 
                index=current_index,
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
    st.subheader("Quick Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("All LFP"):
            st.session_state.cell_configs = ["LFP"] * 8
            st.success("Set all cells to LFP")
            st.rerun()
    
    with col2:
        if st.button("Mixed Config"):
            st.session_state.cell_configs = ["LFP", "NMC", "LFP", "NMC", "LFP", "NMC", "LFP", "NMC"]
            st.success("Set mixed configuration")
            st.rerun()
    
    with col3:
        if st.button("Random Config"):
            st.session_state.cell_configs = [random.choice(["LFP", "NMC", "LTO"]) for _ in range(8)]
            st.success("Set random configuration")
            st.rerun()

# ================================ TASK CONTROLLER PAGE ================================
elif page == "⚡ Task Controller":
    st.header("Battery Task Controller")
    
    # Task configuration
    st.subheader("Configure Task Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        charge_time = st.number_input("Charging Duration (seconds)", min_value=1, max_value=300, value=st.session_state.task_config.get('charge_time', 10))
    
    with col2:
        idle_time = st.number_input("Idle Duration (seconds)", min_value=1, max_value=60, value=st.session_state.task_config.get('idle_time', 5))
    
    with col3:
        discharge_time = st.number_input("Discharge Duration (seconds)", min_value=1, max_value=300, value=st.session_state.task_config.get('discharge_time', 10))
    
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
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Stop Task"):
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
        if st.button("Emergency Stop"):
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
    
    progress_df = pd.DataFrame(progress_data)
    
    if PLOTLY_AVAILABLE:
        fig = px.pie(progress_df, values='Duration', names='Phase', 
                     title="Task Phase Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("Phase Distribution")
        st.bar_chart(progress_df.set_index('Phase')['Duration'])
    
    # Task status
    st.subheader("Current Task Status")
    if st.session_state.task_running:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        current_phase = execute_task()
        if st.session_state.phase_start_time:
            elapsed = time.time() - st.session_state.phase_start_time
            
            if current_phase == "Charging":
                progress = elapsed / charge_time
            elif current_phase == "Idle":
                progress = elapsed / idle_time
            elif current_phase == "Discharging":
                progress = elapsed / discharge_time
            else:
                progress = 1.0
            
            progress = min(progress, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Phase: {current_phase} | Elapsed: {elapsed:.1f}s")
    else:
        st.info("Task not running")

# ================================ DATA ANALYSIS PAGE ================================
elif page == "📈 Data Analysis":
    st.header("Data Analysis Dashboard")
    
    if not st.session_state.historical_data:
        st.warning("No historical data available. Run some tasks first!")
        
        # Generate sample data button
        if st.button("Generate Sample Data for Testing"):
            sample_data = []
            phases = ["Charging", "Idle", "Discharging"]
            for _ in range(50):
                for i in range(8):
                    phase = random.choice(phases)
                    cell_data = generate_cell_data(st.session_state.cell_configs[i], phase, i)
                    if cell_data:
                        sample_data.append(cell_data)
            
            st.session_state.historical_data.extend(sample_data)
            st.success(f"Generated {len(sample_data)} sample records!")
            st.rerun()
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
            time_span = (df_hist['Timestamp'].max() - df_hist['Timestamp'].min()).total_seconds()
            st.metric("Time Span (s)", f"{time_span:.0f}")
        with col4:
            st.metric("Avg Power (W)", f"{df_hist['Power'].mean():.2f}")
        
        # Simple data visualization
        st.subheader("Data Visualization")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_cells = st.multiselect("Select Cells", df_hist['Cell_ID'].unique(), 
                                          default=list(df_hist['Cell_ID'].unique())[:4])
        with col2:
            selected_parameter = st.selectbox("Select Parameter", 
                                            ["Voltage", "Current", "Temperature", "Power", "SOC"])
        
        if selected_cells:
            filtered_df = df_hist[df_hist['Cell_ID'].isin(selected_cells)]
            
            # Simple chart
            if PLOTLY_AVAILABLE:
                fig = px.line(filtered_df, x='Timestamp', y=selected_parameter, 
                             color='Cell_ID', title=f"{selected_parameter} Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Phase distribution
                phase_counts = df_hist['Phase'].value_counts()
                fig2 = px.bar(x=phase_counts.index, y=phase_counts.values, 
                             title="Phase Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.subheader(f"{selected_parameter} Over Time")
                chart_data = filtered_df.pivot_table(
                    index='Timestamp', 
                    columns='Cell_ID', 
                    values=selected_parameter, 
                    aggfunc='mean'
                )
                st.line_chart(chart_data)
        
        # Statistical analysis
        st.subheader("Statistical Summary")
        numeric_cols = ['Voltage', 'Current', 'Temperature', 'Power', 'SOC', 'SOH']
        st.dataframe(df_hist[numeric_cols].describe(), use_container_width=True)

# ================================ DATA MANAGEMENT PAGE ================================
elif page == "📝 Data Management":
    st.header("Data Management")
    
    # Current session data
    st.subheader("Current Session Data")
    if st.session_state.historical_data:
        df_session = pd.DataFrame(st.session_state.historical_data)
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records in Memory", len(df_session))
        with col2:
            memory_size = df_session.memory_usage(deep=True).sum() / 1024  # KB
            st.metric("Memory Usage (KB)", f"{memory_size:.1f}")
        with col3:
            if df_session['Timestamp'].dtype == 'object':
                df_session['Timestamp'] = pd.to_datetime(df_session['Timestamp'])
            duration = (df_session['Timestamp'].max() - df_session['Timestamp'].min()).total_seconds()
            st.metric("Data Duration (s)", f"{duration:.0f}")
        
        # Show recent data
        st.subheader("Recent Data (Last 20 Records)")
        st.dataframe(df_session.tail(20), use_container_width=True)
        
        # Data export options
        st.subheader("Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save to CSV File"):
                filename = save_to_csv(st.session_state.historical_data)
                if filename:
                    st.success(f"Data saved to {filename}")
        
        with col2:
            csv_data = df_session.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"battery_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            json_data = df_session.to_json(orient='records', indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"battery_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Data management actions
        st.subheader("Data Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear All Data"):
                st.session_state.historical_data = []
                st.success("All data cleared!")
                st.rerun()
        
        with col2:
            data_limit = st.number_input("Data Limit", min_value=100, max_value=5000, value=1000)
            if st.button("Apply Limit"):
                if len(st.session_state.historical_data) > data_limit:
                    st.session_state.historical_data = st.session_state.historical_data[-data_limit:]
                    st.info(f"Data limited to last {data_limit} records")
                    st.rerun()
        
        with col3:
            if st.button("Remove Old Data"):
                # Remove data older than 1 hour
                current_time = datetime.datetime.now()
                one_hour_ago = current_time - datetime.timedelta(hours=1)
                
                filtered_data = []
                for record in st.session_state.historical_data:
                    record_time = datetime.datetime.fromisoformat(record['Timestamp'])
                    if record_time > one_hour_ago:
                        filtered_data.append(record)
                
                removed_count = len(st.session_state.historical_data) - len(filtered_data)
                st.session_state.historical_data = filtered_data
                st.info(f"Removed {removed_count} old records")
                if removed_count > 0:
                    st.rerun()
    
    else:
        st.info("No data available in current session.")
        st.write("To generate data:")
        st.write("1. Go to Cell Configuration and set up your cells")
        st.write("2. Go to Task Controller and start a task")
        st.write("3. Monitor in Real-Time Monitoring page")

# Footer with system info
st.sidebar.markdown("---")
st.sidebar.info("Enhanced BMS v2.1 - Fixed & Optimized")

if st.sidebar.button("System Info"):
    st.sidebar.write(f"Python modules available:")
    st.sidebar.write(f"- Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    st.sidebar.write(f"- Serial: {'✅' if SERIAL_AVAILABLE else '❌'}")
    st.sidebar.write(f"- Records in memory: {len(st.session_state.historical_data)}")
    st.sidebar.write(f"- Task running: {'✅' if st.session_state.task_running else '❌'}")

# Add error boundary
try:
    # Main execution happens above
    pass
except Exception as e:
    st.error(f"Application Error: {e}")
    st.info("Please refresh the page or contact support if the problem persists.")
    
    # Debug info
    if st.checkbox("Show Debug Info"):
        st.code(f"Error: {str(e)}")
        st.code(f"Session State Keys: {list(st.session_state.keys())}")
