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
import torch
from torch import nn
from model import EngineLSTM, predict

# Load or initialize configuration
config_file = "config.json"
default_config = {
    "log_path": "data/",
    "update_interval": 1000,
    
    "input_sequence_length": 80,
    "output_sequence_length": 20,
    "hidden_size": 64,
    "num_layers": 2,
    
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 0.001
}

# Load configuration
def load_config():
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Config not found")
        logging.info("Saving default config")
        save_config(default_config)
        return default_config

# Save configuration
def save_config(new_config):
    with open(config_file, "w") as f:
        json.dump(new_config, f, indent=4)

# Load the configuration
config = load_config()

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Vehicle Condition Monitoring"

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
        
        data = pd.read_csv(csv_file,skiprows=1,dtype=np.float32)
        return data.tail(config['input_sequence_length'])
        
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
            html.Label("Update Interval: "),
            dcc.Input(id="update_interval", type="number", value=config["update_interval"], step=50)
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Input Sequence Length: "),
            dcc.Input(id="input_sequence_length", type="number", value=config["input_sequence_length"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Output Sequence Length: "),
            dcc.Input(id="output_sequence_length", type="number", value=config["output_sequence_length"])
        ], style={"marginBottom": "5px"}),
        
        html.Div([
            html.Label("Hidden Layer Size: "),
            dcc.Input(id="hidden_size", type="number", value=config["hidden_size"])
        ], style={"marginBottom": "5px"}),

        html.Div([
            html.Label("Number of Layers: "),
            dcc.Input(id="num_layers", type="number", value=config["num_layers"])
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

# Fetch pretrained model files    
models = {}
for output in output_keys[1:]:
    model_name = output.split('(')[0].strip().replace(' ','_')
    
    model = EngineLSTM(input_size=1, hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_sequence_length'])
    model.load_state_dict(torch.load(f"models/{model_name}_model.pth", weights_only=False))    

    models[output] = model

@app.callback(
    outputs,
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch data
    data = get_data()
    if data.empty:
        logging.info("No data fetched")
        return [None] * len(outputs)

    # Log fetched data in the terminal
    logging.info(f"Fetched data: {data}")
    
    figures = []
    time_column = data.columns[0]  # Assume the first column is time
    time_data = data[time_column]
    for key in data.columns[1:]:
        model = models[key]
        predictions = predict(data[key].values, model) # Predict future values
        predictions = np.insert(predictions, 0, data[key].iloc[-1]) # Set the first point to the last real datapoint

        # Create future time values for predictions         
        time_intervals = time_data.diff().iloc[1:]  # Calculate intervals
        mean_time_per_point = np.mean(time_intervals)
        pred_time_start = time_data.iloc[-1]
        pred_time_end = pred_time_start + config["output_sequence_length"] * mean_time_per_point
        predictions_time = np.linspace(pred_time_start, pred_time_end, len(predictions))
        
        # Create figures
        figures.append({
            'data': [
                go.Scatter(x=time_data,y=data[key],mode='lines+markers',name=f'{key} Inputs'),
                go.Scatter(x=predictions_time,y=predictions,mode='lines+markers',name=f'{key} Predictions')
            ],
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
        # Dashboard Parameters
        State("log_path", "value"),
        State("update_interval", "value"),
        
        # Model Parameters
        State("input_sequence_length", "value"),
        State("output_sequence_length", "value"),
        State("hidden_size", "value"),
        State("num_layers", "value"),

        # Training parameters
        State("batch_size", "value"),
        State("num_epochs", "value"),
        State("learning_rate", "value"),
        
        
    ]
)
def save_configuration(n_clicks, log_path, update_interval, input_seq_len, output_seq_len, hidden_size, num_layers, batch_size, num_epochs, learning_rate):
    if n_clicks:
        new_config = {
            "log_path": log_path,
            "update_interval": update_interval,
            "input_sequence_length": input_seq_len,
            "output_sequence_length": output_seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate
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
