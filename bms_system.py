import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime
import json

try:
    import serial
except ImportError:
    serial = None

st.set_page_config(page_title="Battery Management System UI", layout="wide")
st.title("🔋 Battery Management System - Unified Dashboard")

page = st.sidebar.selectbox("Choose Dashboard", [
    "📊 Main Monitoring",
    "🛠️ User Controls",
    "📝 Data Logging",
    "📡 Real-Time Sensor",
    "🔧 Configure Cells"
])

# ------------------------------ Function for random cell data ------------------------------
def generate_random_cell_data(cell_type="LFP"):
    voltage = round(random.uniform(3.0, 3.6), 3) if cell_type == "LFP" else round(random.uniform(3.3, 4.2), 3)
    return {
        "Voltage (V)": voltage,
        "Current (A)": round(random.uniform(-5.0, 5.0), 2),
        "Temperature (°C)": round(random.uniform(25.0, 45.0), 1),
        "Capacity (Ah)": round(random.uniform(2.0, 3.0), 2),
        "State of Charge (%)": round(random.uniform(40, 100), 1),
        "State of Health (%)": round(random.uniform(85, 100), 1),
        "Internal Resistance (mΩ)": round(random.uniform(10, 30), 1),
        "Balancing": random.choice(["Yes", "No"]),
        "Protection Alert": random.choice(["None", "Overvoltage", "Undervoltage", "Overtemp", "Short Circuit"]),
    }

# ------------------------------ Main Monitoring Page ------------------------------
if page == "📊 Main Monitoring":
    cell_type = st.sidebar.selectbox("Select Cell Type", ["LFP", "NMC"])
    cycle_count = st.sidebar.number_input("Enter Battery Cycle Count", min_value=0, max_value=10000, value=120)
    cell_data = [generate_random_cell_data(cell_type) for _ in range(8)]
    df = pd.DataFrame(cell_data)
    df.index = [f"Cell {i+1}" for i in range(8)]

    st.subheader("📊 Individual Cell Parameters")
    st.dataframe(df.style.highlight_max(axis=0))

    total_voltage = df["Voltage (V)"].sum()
    total_current = df["Current (A)"].sum()
    total_power = round(total_voltage * total_current, 2)
    average_temp = df["Temperature (°C)"].mean()
    average_soc = df["State of Charge (%)"].mean()
    average_soh = df["State of Health (%)"].mean()

    st.subheader("📦 Battery Pack Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Voltage (V)", f"{total_voltage:.2f}")
    col2.metric("Total Current (A)", f"{total_current:.2f}")
    col3.metric("Total Power (W)", f"{total_power:.2f}")
    col4.metric("Cycle Count", f"{cycle_count}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Avg Temperature (°C)", f"{average_temp:.1f}")
    col6.metric("Avg SoC (%)", f"{average_soc:.1f}")
    col7.metric("Avg SoH (%)", f"{average_soh:.1f}")

    st.subheader("📈 Voltage Distribution")
    st.bar_chart(df["Voltage (V)"])
    st.subheader("🌡️ Temperature Trends")
    st.line_chart(df["Temperature (°C)"])
    st.subheader("⚡ Current Flow Per Cell")
    st.bar_chart(df["Current (A)"])

    st.subheader("🚨 Protection & Safety Alerts")
    alerts = df[df["Protection Alert"] != "None"]
    if alerts.empty:
        st.success("✅ No Protection Alerts Detected")
    else:
        st.warning("⚠️ Some cells have active protection alerts:")
        st.table(alerts[["Protection Alert"]])

# ------------------------------ User Controls Page ------------------------------
elif page == "🛠️ User Controls":
    st.subheader("⚡ Control Battery Operation")
    if st.button("Start Charging"):
        st.success("✅ Charging started...")
    if st.button("Stop Charging"):
        st.warning("⚠️ Charging stopped.")
    if st.button("Start Discharging"):
        st.success("✅ Discharging started...")
    if st.button("Stop Discharging"):
        st.warning("⚠️ Discharging stopped.")

    st.subheader("⚙️ Settings")
    balance_cells = st.checkbox("Enable Cell Balancing")
    voltage_limit = st.slider("Set Overvoltage Threshold", 3.0, 4.5, 4.2)
    selected_cells = st.multiselect("Select Cells to Monitor", [f"Cell {i+1}" for i in range(8)], default=["Cell 1"])
    st.info("These controls are virtual and connect to your BMS logic in real-time system.")

# ------------------------------ Data Logging Page ------------------------------
elif page == "📝 Data Logging":
    st.subheader("📁 Data Logging to CSV")
    simulated_data = {
        "Cell": [f"Cell {i+1}" for i in range(8)],
        "Voltage": [round(3.2 + i * 0.05, 3) for i in range(8)],
        "Temperature": [round(30 + i, 1) for i in range(8)],
        "Current": [round(0.5 + i * 0.1, 2) for i in range(8)],
    }
    df = pd.DataFrame(simulated_data)
    st.dataframe(df)
    log_file = "battery_log.csv"
    if st.button("📩 Save This Snapshot"):
        df["Timestamp"] = datetime.datetime.now().isoformat()
        df.to_csv(log_file, mode='a', index=False, header=True)
        st.success(f"Logged successfully to `{log_file}`")

# ------------------------------ Real-Time Sensor Page ------------------------------
elif page == "📡 Real-Time Sensor":
    st.subheader("📡 Real-Time Battery Sensor Feed")
    if serial is None:
        st.error("PySerial not installed. Use `pip install pyserial` to enable.")
    else:
        try:
            ser = serial.Serial('COM3', 9600, timeout=2)
            line = ser.readline().decode('utf-8').strip()
            sensor_data = json.loads(line)
            if "cells" in sensor_data:
                df = pd.DataFrame(sensor_data["cells"])
                df.index = [f"Cell {i+1}" for i in range(len(df))]
                st.dataframe(df)
            else:
                st.warning("⚠️ No cell data received.")
        except Exception as e:
            st.error(f"Serial Error: {e}")

# ------------------------------ Configure Cells Page (Your Logic Optimized) ------------------------------
elif page == "🔧 Configure Cells":
    st.subheader("🔧 Define Each Cell Type and View Data")
    list_of_cell = []
    cols = st.columns(8)
    for i in range(8):
        with cols[i]:
            cell_type = st.selectbox(f"Cell {i+1}", ["lfp", "nmc"], key=f"cell_{i}")
            list_of_cell.append(cell_type)

    cells_data = {}
    for idx, cell_type in enumerate(list_of_cell, start=1):
        cell_key = f"Cell_{idx}_{cell_type.upper()}"
        voltage = 3.2 if cell_type == "lfp" else 3.6
        current = round(random.uniform(0.0, 5.0), 2)
        temp = round(random.uniform(25.0, 40.0), 1)
        capacity = round(voltage * current, 2)
        cells_data[cell_key] = {
            "Voltage (V)": voltage,
            "Current (A)": current,
            "Temperature (°C)": temp,
            "Capacity (Wh)": capacity
        }

    df_cells = pd.DataFrame.from_dict(cells_data, orient="index")
    df_cells.index.name = "Cell"
    st.dataframe(df_cells)

    csv = df_cells.to_csv().encode("utf-8")
    st.download_button("📥 Download Cell Data", csv, "cell_data.csv", "text/csv")
