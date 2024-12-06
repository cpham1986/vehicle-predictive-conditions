import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import plotly.graph_objs as go
from collections import deque
import logging

# Load or initialize configuration
config_file = "config.json"
default_config = {
    "log_path": "data/",
    "input_sequence_length": 50,
    "output_sequence_length": 50,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "num_points": 20,
    "update_interval": 1000
}

# Load configuration
def load_config():
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("config not found")
        return default_config

# Save configuration
def save_config(new_config):
    with open(config_file, "w") as f:
        json.dump(new_config, f, indent=4)

# Load the configuration
config = load_config()

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "System Monitoring"

# App Layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# History for main dashboard (deque for fixed-size storage)
history = {
    "time": deque(maxlen=config["num_points"]),
    "control_module_voltage": deque(maxlen=config["num_points"]),
    "engine_rpm": deque(maxlen=config["num_points"]),
    "engine_coolant_temp": deque(maxlen=config["num_points"])
}

# Headers for output
headers = {
    "time": "Time",
    "control_module_voltage": "Control Module Voltage",
    "engine_rpm": "Engine RPM",
    "engine_coolant_temp": "Engine Coolant Temp"    
}

# Units for output
units = {
    "time": "s",
    "control_module_voltage": "V",
    "engine_rpm": "RPM",
    "engine_coolant_temp": "degrees C"    
}

# Find the most recent CSV file
def get_most_recent_csv(directory):
    try:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not csv_files:
            return None
        return max((os.path.join(directory, f) for f in csv_files), key=os.path.getmtime)
    except Exception as e:
        return None

# Fetch the latest data
def get_data():
    try:
        csv_file = get_most_recent_csv(config["log_path"])
        if not csv_file:
            return {}
        with open(csv_file, "rb") as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode().strip()
        data = last_line.split(",")
        return {
            "time": float(data[0]),
            "control_module_voltage": float(data[2]),
            "engine_rpm": float(data[3]),
            "engine_coolant_temp": float(data[4])
        }
    except Exception as e:
        return {}

# Configuration Page Layout
def config_page():
    return html.Div([
        html.H1("Configuration"),

        html.Div([
            html.Label("Log Path: "),
            dcc.Input(id="log_path", type="text", value=config["log_path"],
            style={"width": "500px"})
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Input Sequence Length: "),
            dcc.Input(id="input_sequence_length", type="number", value=config["input_sequence_length"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Output Sequence Length: "),
            dcc.Input(id="output_sequence_length", type="number", value=config["output_sequence_length"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Batch Size: "),
            dcc.Input(id="batch_size", type="number", value=config["batch_size"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Number of Epochs: "),
            dcc.Input(id="num_epochs", type="number", value=config["num_epochs"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Learning Rate: "),
            dcc.Input(id="learning_rate", type="number", value=config["learning_rate"], step=0.0001)
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Number of Points: "),
            dcc.Input(id="num_points", type="number", value=config["num_points"])
        ], style={"marginBottom": "10px"}),
        
        html.Div([
            html.Label("Update Interval: "),
            dcc.Input(id="update_interval", type="number", value=config["update_interval"], step=50)
        ], style={"marginBottom": "10px"}),

        html.Button("Save Configuration", id="save_config", style={"marginTop": "20px"}),

        html.Div(id="save_feedback", style={"marginTop": "10px"}),

        html.Br(),

        dcc.Link("Go to Dashboard", href="/", style={"marginTop": "20px"})
    ])


# Main Dashboard Layout
def dashboard_page():
    # Generate Line Charts and outputs
    graphs = []
    for key in history.keys():
        if key != 'time':
            graphs.append(dcc.Graph(id=f'{key}-usage-graph'))

    return html.Div([
        html.H1('System Monitoring Dashboard'),

        *graphs,

        # Interval for updating the dashboard
        dcc.Interval(
            id='interval-component',
            interval=config['update_interval'],
            n_intervals=0
        ),
        dcc.Link("Go to Configuration", href="/config")
    ])


# Update Graphs
outputs = []
for key in history.keys():
    if key != 'time':
        outputs.append(Output(f'{key}-usage-graph', 'figure'))
@app.callback(
    outputs,
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch system stats (RAM, CPU, and Disk)
    data = get_data()

    if not data:
        logging.info("No data fetched")
        return [None] * len(history.keys())

    # Log fetched data in the terminal
    logging.info(f"Fetched data: {data}")

    figures = []
    for key in history.keys():
        history[key].append(data[key])
        
        # Create Line Charts
        if key != 'time':
            figures.append({
                'data': [go.Scatter(
                    x=list(history['time']),
                    y=list(history[key]),
                    mode='lines+markers',
                    name=f'{key}'
                )],
                'layout': go.Layout(
                    title=f'{headers[key]}',
                    xaxis=dict(title='Time (s)', tickformat='%H:%M:%S'),  # Format the time
                    yaxis=dict(title=f'{units[key]}'),
                )
            })
    return figures

# Save Configuration
@app.callback(
    Output("save_feedback", "children"),
    [Input("save_config", "n_clicks")],
    [
        State("log_path", "value"),
        State("input_sequence_length", "value"),
        State("output_sequence_length", "value"),
        State("batch_size", "value"),
        State("num_epochs", "value"),
        State("learning_rate", "value"),
        State("num_points", "value"),
        State("update_interval", "value")
    ]
)
def save_configuration(n_clicks, log_path, input_seq_len, output_seq_len, batch_size, num_epochs, learning_rate, num_points, update_interval):
    if n_clicks:
        new_config = {
            "log_path": log_path,
            "input_sequence_length": input_seq_len,
            "output_sequence_length": output_seq_len,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_points": num_points,
            "update_interval": update_interval
        }
        save_config(new_config)
        config = new_config
        return "Configuration saved successfully!\nRestart the server to see the changes."
    return ""

# URL Routing
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/config":
        return config_page()
    else:
        return dashboard_page()

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
