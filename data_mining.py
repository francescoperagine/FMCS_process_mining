import os
import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter

# Evaluation metrics
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness
from pm4py.algo.evaluation.precision import algorithm as precision
from pm4py.algo.evaluation.simplicity import algorithm as simplicity
from pm4py.algo.evaluation.generalization import algorithm as generalization

# text_to_csv.py
import csv

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

def txt_to_xes(input_file, output_file):
    """
    Converts a text file to an XES event log format for process mining.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output XES file.
    """
    # Initialize an empty event log
    log = EventLog()

    # Read the text file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Group events by trace (example: one trace for simplicity)
    trace = Trace()  # Create a single trace
    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        timestamp = parts[0] + " " + parts[1]
        location = parts[2]
        state = parts[3]
        activity = parts[4]

        # Create an event
        event = Event({
            "concept:name": activity,
            "time:timestamp": timestamp,
            "org:resource": location,
            "state": state,
            "case:concept:name": f"Case_{idx}"  # Assign a case ID
        })

        # Add event to the trace
        trace.append(event)

    # Add the trace to the log
    log.append(trace)

    # Export the event log to XES
    xes_exporter.apply(log, output_file)
    print(f"Data successfully converted to {output_file}")

def load_and_process_xes(file_path):
    """
    Load and process an XES log file into a pandas DataFrame and structured event log.
    """
    # Load the XES log using PM4Py
    event_log = xes_importer.apply(file_path)

    # Convert the XES event log to a pandas DataFrame
    df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    df['time:timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Process the DataFrame
    # Simplify activity label
    df['activity_label'] = df['org:resource'] + '_' + df['state']
    
    # Create cases based on hour windows
    df['hour'] = pd.to_datetime(df['time:timestamp']).dt.strftime('%Y-%m-%d_%H')
    df['case:concept:name'] = 'hour_' + df['hour']
    
    # Sort by timestamp
    df = df.sort_values('time:timestamp')

    # Prepare for PM4Py (if needed again later)
    event_log = log_converter.apply(df)
    
    return df, event_log

def calculate_metrics(event_log, net, initial_marking, final_marking):
    """Calculate all four metrics for a given model"""
    metrics = {}
    
    # Fitness
    fitness_value = fitness.apply(event_log, net, initial_marking, final_marking, variant=fitness.Variants.TOKEN_BASED)
    metrics['fitness'] = fitness_value['average_trace_fitness']
    
    # Precision
    metrics['precision'] = precision.apply(event_log, net, initial_marking, final_marking, variant=precision.Variants.ETCONFORMANCE_TOKEN)
    
    # Simplicity
    metrics['simplicity'] = simplicity.apply(net)
    
    # Generalization
    metrics['generalization'] = generalization.apply(event_log, net, initial_marking, final_marking)
    
    return metrics

def apply_miners(event_log):
    """Apply all three miners and collect their metrics"""
    results = {}
    
    # Alpha Miner
    print("\nApplying Alpha Miner...")
    net, initial_marking, final_marking = alpha_miner.apply(event_log)
    results['Alpha'] = {
        'model': (net, initial_marking, final_marking),
        'metrics': calculate_metrics(event_log, net, initial_marking, final_marking)
    }

    # Heuristics Miner
    print("\nApplying Heuristics Miner...")
    net, initial_marking, final_marking = heuristics_miner.apply(event_log)
    results['Heuristics'] = {
        'model': (net, initial_marking, final_marking),
        'metrics': calculate_metrics(event_log, net, initial_marking, final_marking)
    }
    
    # Inductive Miner
    print("\nApplying Inductive Miner...")
    tree = inductive_miner.apply(event_log)
    net, initial_marking, final_marking = pt_converter.apply(tree)
    results['Inductive'] = {
        'model': (net, initial_marking, final_marking),
        'metrics': calculate_metrics(event_log, net, initial_marking, final_marking)
    }
    
    return results

def display_comparison(results):
    """Display a comparison of all metrics for all miners"""
    print("\nMetrics Comparison:")
    print("-" * 50)
    print(f"{'Miner':<12} {'Fitness':>8} {'Precision':>10} {'Simplicity':>10} {'Gen.':>8}")
    print("-" * 50)
    
    for miner_name, miner_results in results.items():
        metrics_values = miner_results['metrics']
        print(f"{miner_name:<12} "
              f"{metrics_values['fitness']:8.3f} "
              f"{metrics_values['precision']:10.3f} "
              f"{metrics_values['simplicity']:10.3f} "
              f"{metrics_values['generalization']:8.3f}")
    print("-" * 50)

def save_visualizations(results):
    """Save visualizations for all miners"""

    # Create resources/figures directory if it doesn't exist
    output_dir = os.path.join('resources', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    for miner_name, miner_results in results.items():
        net, initial_marking, final_marking = miner_results['model']
        gviz = pn_visualizer.apply(
            net, initial_marking, final_marking,
            parameters={
                pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png",
                "rankdir": "LR"  # Left to right layout
            }
        )
        output_path = os.path.join(output_dir, f"petri_net_{miner_name.lower()}.png")
        pn_visualizer.save(gviz, output_path)
        print(f"Saved visualization for {miner_name} Miner to: {output_path}")

def main():

    # File paths
    input_file = os.path.join('data', 'raw', 'tm001.txt')
    output_file = os.path.join('data', 'processed', 'dataset.xes')
    figures_path = os.path.join('resources', 'figures')

    # Convert to XES
    txt_to_xes(input_file, output_file)
    
    # Process the log
    print("Loading and processing log file...")
    df, event_log = load_and_process_xes(output_file)
    print(df.head())

    # Count the number of events per case
    case_counts = df['case:concept:name'].value_counts()
    print(case_counts)

    # Plot activity frequencies
    df['concept:name'].value_counts().plot(kind='bar')

    # Apply miners and get results
    results = apply_miners(event_log)
    
    # Display comparison
    display_comparison(results)
    
    # Save visualizations
    save_visualizations(results, figures_path)

if __name__ == "__main__":
    main()