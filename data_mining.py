import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.statistics.start_activities.log import get as start_activities
from pm4py.statistics.end_activities.log import get as end_activities
from pm4py.objects.petri_net.utils import reachability_graph

from sklearn.preprocessing import KBinsDiscretizer

class ProcessLogPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.discretizers = {}
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self
        
    def clean_data(self):
        """Use data-driven discretization"""
        if self.df is None:
            self.load_data()
            
        # Keep essential features that show clear patterns
        columns_to_keep = ['accidents', 'traffic_density', 'rain_intensity', 'time_of_day']
        self.df = self.df[columns_to_keep]
        
        # Create discretizers for numeric columns
        for col in ['accidents', 'traffic_density', 'rain_intensity']:
            self.discretizers[col] = KBinsDiscretizer(
                n_bins=4,  # Create quartile-based bins
                encode='ordinal',
                strategy='quantile'
            )
            self.df[f'{col}_bin'] = self.discretizers[col].fit_transform(
                self.df[[col]]
            ).astype(int)
            
        # Create meaningful time periods (6-hour windows)
        self.df['time_period'] = pd.cut(
            self.df['time_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['NIGHT', 'MORNING', 'AFTERNOON', 'EVENING']
        )
        
        return self
        
    def create_activity_labels(self):
        """Create activity labels based on statistical patterns"""
        def get_activity_label(row):
            # Get severity level based on accidents quartile
            severity_map = {
                0: 'LOW',
                1: 'MEDIUM_LOW',
                2: 'MEDIUM_HIGH',
                3: 'HIGH'
            }
            severity = severity_map[row['accidents_bin']]
            
            # Combine conditions only when they're significant
            conditions = []
            
            # Add traffic condition if it's in upper two quartiles
            if row['traffic_density_bin'] >= 2:
                traffic_level = 'HIGH' if row['traffic_density_bin'] == 3 else 'MODERATE'
                conditions.append(f'TRAFFIC_{traffic_level}')
            
            # Add rain condition if it's in upper two quartiles
            if row['rain_intensity_bin'] >= 2:
                rain_level = 'HEAVY' if row['rain_intensity_bin'] == 3 else 'MODERATE'
                conditions.append(f'RAIN_{rain_level}')
            
            # Add time period always
            conditions.append(str(row['time_period']))
            
            # Combine into final activity label
            return f'ACCIDENTS_{severity}_' + '_'.join(conditions)
        
        self.df['activity'] = self.df.apply(get_activity_label, axis=1)
        return self
        
    def prepare_event_log(self):
        """Create cases based on time sequences with proper filtering"""
        # Create 6-hour cases aligned with time periods
        self.df['time_block'] = (self.df['time_of_day'] // 6).astype(int)
        self.df['case:concept:name'] = 'case_' + self.df['time_block'].astype(str)
        self.df['concept:name'] = self.df['activity']
        self.df['time:timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(self.df['time_of_day'], unit='h')
        
        # Filter based on activity sequences, not just frequency
        sequence_counts = self.df.groupby('case:concept:name')['activity'].agg(list).apply(tuple).value_counts()
        common_sequences = sequence_counts[sequence_counts >= 2].index
        
        # Keep only cases that follow common sequences
        valid_cases = self.df.groupby('case:concept:name')['activity'].agg(tuple).isin(common_sequences)
        valid_cases = valid_cases[valid_cases].index
        self.df = self.df[self.df['case:concept:name'].isin(valid_cases)]
        
        # Create event log
        event_log = log_converter.apply(self.df)
        return event_log
    
class ProcessMiner:
    def __init__(self, event_log):
        self.event_log = event_log
        self.net = None
        self.initial_marking = None
        self.final_marking = None
        
    def calculate_quality_metrics(self):
        """Calculate metrics using PM4Py's built-in methods"""
        from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
        from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
        from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
        from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

        metrics = {}
        
        try:
            # Fitness
            fitness = replay_fitness.apply(self.event_log, self.net, 
                                        self.initial_marking, self.final_marking,
                                        variant=replay_fitness.Variants.TOKEN_BASED)
            metrics['fitness'] = fitness['average_trace_fitness']
            
            # Precision - using ETConformance
            metrics['precision'] = precision_evaluator.apply(self.event_log, self.net,
                                                        self.initial_marking, 
                                                        self.final_marking,
                                                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
            
            # Generalization
            metrics['generalization'] = generalization_evaluator.apply(self.event_log, 
                                                                    self.net,
                                                                    self.initial_marking,
                                                                    self.final_marking)
            
            # Simplicity
            metrics['simplicity'] = simplicity_evaluator.apply(self.net)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            metrics = {
                'fitness': 0.0,
                'precision': 0.0,
                'generalization': 0.0,
                'simplicity': 0.0
            }
        
        return metrics
        
    def save_visualization(self, path, filename):
        if not all([self.net, self.initial_marking, self.final_marking]):
            raise ValueError("Model not yet mined")
            
        gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking,
                                parameters={
                                    pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png",
                                    "show_labels": True
                                })
        pn_visualizer.save(gviz, os.path.join(path, filename))

class AlphaMiner(ProcessMiner):
    def mine(self):
        """Apply Alpha miner"""
        self.net, self.initial_marking, self.final_marking = alpha_miner.apply(self.event_log)
        return self

class InductiveMiner(ProcessMiner):
    def mine(self):
        """Apply Inductive miner"""
        tree = inductive_miner.apply(self.event_log)
        self.net, self.initial_marking, self.final_marking = pt_converter.apply(tree)
        return self

class ILPMiner(ProcessMiner):
    def mine(self):
        """Apply ILP miner"""
        self.net, self.initial_marking, self.final_marking = ilp_miner.apply(self.event_log)
        return self
    
def plot_metrics_comparison(all_metrics):
    """Plot the four quality metrics for each miner"""
    plt.figure(figsize=(12, 6))
    
    metrics = ['fitness', 'precision', 'generalization', 'simplicity']
    miners = list(all_metrics.keys())
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (miner, miner_metrics) in enumerate(all_metrics.items()):
        values = [miner_metrics[metric] for metric in metrics]
        offset = width * i
        plt.bar(x + offset, values, width, label=miner)
    
    plt.ylabel('Score')
    plt.title('Process Mining Quality Metrics')
    plt.xticks(x + width, metrics, rotation=45)
    plt.legend()
    
    plt.ylim(0, 1)
    plt.tight_layout()
    return plt.gcf()

def main():
    data_path = os.path.join('data', 'raw', 'traffic_accidents', 'traffic_accidents.csv')
    image_path = os.path.join('resources', 'figures')
    os.makedirs(image_path, exist_ok=True)
    
    # Preprocess data
    preprocessor = ProcessLogPreprocessor(data_path)
    event_log = (preprocessor.load_data()
                            .clean_data()
                            .create_activity_labels()
                            .prepare_event_log())
    
    # Print initial log statistics
    print("\nLog Statistics:")
    variants = case_statistics.get_variant_statistics(event_log)
    print(f"Number of cases: {len(event_log)}")
    print(f"Number of variants: {len(variants)}")
    print(f"Number of unique activities: {len(set(trace[0]['concept:name'] for trace in event_log))}")
    
    # Initialize miners
    miners_dict = {
        'Alpha': AlphaMiner(event_log),
        'Inductive': InductiveMiner(event_log),
        'ILP': ILPMiner(event_log)
    }
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Apply mining and collect metrics
    for name, miner in miners_dict.items():
        print(f"\nApplying {name} Miner...")
        miner.mine()
        
        # Calculate quality metrics
        quality_metrics = miner.calculate_quality_metrics()
        all_metrics[name] = quality_metrics
        
        # Save visualization
        miner.save_visualization(image_path, f"petri_net_{name.lower()}.png")
        
        # Print metrics and properties
        print(f"\n{name} Miner Metrics:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.3f}")
            
        print(f"\n{name} Miner Properties:")
        print(f"Places: {len(miner.net.places)}")
        print(f"Named Transitions: {sum(1 for t in miner.net.transitions if t.label is not None)}")
        print(f"Silent Transitions: {sum(1 for t in miner.net.transitions if t.label is None)}")
    
    # Plot comparative metrics
    fig = plot_metrics_comparison(all_metrics)
    plt.savefig(os.path.join(image_path, 'metrics_comparison.png'))
    plt.close()

if __name__ == "__main__":
    main()