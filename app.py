import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import os
import plotly.graph_objs as go
from collections import deque
import logging
import numpy as np
import pandas as pd

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
        logging.error("config not found")
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

# Find the most recent CSV file
def get_most_recent_csv(directory):
    try:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not csv_files:
            return None
        return max((os.path.join(directory, f) for f in csv_files), key=os.path.getmtime)
    except Exception as e:
        return None

def get_data():
    try:
        csv_file = get_most_recent_csv(config["log_path"])
        if not csv_file:
            return []
        
        '''
        with open(csv_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            end_position = f.tell()
            buffer_size = 1024
            lines = []
            while len(lines) <= 50 and f.tell() > 0:
                f.seek(max(f.tell() - buffer_size, 0), os.SEEK_SET)
                chunk = f.read(min(buffer_size, f.tell()))
                lines = chunk.split(b"\n") + lines
                f.seek(max(f.tell() - buffer_size, 0), os.SEEK_SET)
        '''
        data = pd.read_csv(csv_file,skiprows=1,dtype=np.float32)
        return data.tail(config['num_points'])
        

        #    last_50_lines = [line.decode().strip() for line in lines if line.strip()][-50:]
        '''
        parsed_data = {"time":[],"control_module_voltage":[],"engine_rpm":[],"engine_coolant_temp":[]}
        for line in last_50_lines:
            data = line.split(",")
            parsed_data["time"].append(float(data[0]))
            parsed_data["control_module_voltage"].append(float(data[2]))
            parsed_data["engine_rpm"].append(float(data[3]))
            parsed_data["engine_coolant_temp"].append(float(data[4]))
#        print(parsed_data)
        return parsed_data
        '''

    except Exception as e:
        logging.error(e)
        return []


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
graphs = []
outputs = []
csv_file = get_most_recent_csv(config["log_path"])
if not csv_file:
    exit()
output_keys = pd.read_csv(csv_file,skiprows=1).keys()
for key in output_keys:
    if key == output_keys[0]: continue
    graphs.append(dcc.Graph(id=f'{key}-usage-graph'))
    outputs.append(Output(f'{key}-usage-graph', 'figure'))
@app.callback(
    outputs,
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch system stats (RAM, CPU, and Disk)
    data = get_data()

    if data.empty:
        logging.info("No data fetched")
        return [None] * len(outputs)

    # Log fetched data in the terminal
    logging.info(f"Fetched data: {data}")

    figures = []
    time_column = data.columns[0]  # Assume the first column is time
    for key in data.columns[1:]:
        figures.append({
            'data': [go.Scatter(
                x=data[time_column],
                y=data[key],
                mode='lines+markers',
                name=key
            )],
            'layout': go.Layout(
                title=key,
                xaxis=dict(title='Time', tickformat='%H:%M:%S'),
                yaxis=dict(title=key)
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
