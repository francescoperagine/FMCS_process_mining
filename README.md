# Process Mining Analysis of Activities of Daily Living

This project is part of the Formal Methods for Computer Science Master's Degree course at the University of Bari Aldo Moro (UniBA). It focuses on applying process mining techniques to analyze Activities of Daily Living (ADL) data from smart home environments.

## Dataset

The project uses the TM001 Single-resident home dataset from the CASAS Smart Home Project (Washington State University). The dataset contains sensor readings and activity annotations from a single-resident smart home environment.

Example data format:
```
2016-11-12 21:48:36.106538 KitchenARefrigerator ON Other_Activity
2016-11-12 21:48:36.445749 KitchenASink OFF Other_Activity
2016-11-12 21:48:43.602530 KitchenARefrigerator OFF Other_Activity
```

Each line contains:
- Timestamp (date and time)
- Sensor ID
- Sensor State (ON/OFF)
- Activity Label

## Features

The project implements several process mining techniques:
- Alpha Miner
- Heuristic Miner
- Inductive Miner

Key functionalities include:
- Data preprocessing and event log creation
- Process model discovery using multiple algorithms
- Performance metrics calculation (Fitness, Precision, Generalization, Simplicity)
- Petri net visualization
- User behavior pattern analysis

## Project Structure

- `data/raw/` - Contains the raw dataset files
- `data/processed/user_habits_analysis.txt` - Processed output file with analyzed behavior patterns
- `resources/figures/` - Generated visualizations and Petri nets
- `process_mining.ipynb` - Main Jupyter notebook containing the analysis

## Dependencies

- pm4py - Process Mining library
- pandas - Data manipulation
- matplotlib - Visualization
- sklearn - Data preprocessing

## Metrics and Analysis

The project evaluates process models using four key metrics:
- Fitness: How well the model can replay the observed behavior
- Precision: How well the model avoids allowing unobserved behavior
- Generalization: The model's ability to handle unseen cases
- Simplicity: The structural complexity of the model

## Usage

1. Place the TM001 dataset in the `data/raw/` directory
2. Run the cells in `process_mining.ipynb`
3. Generated visualizations will be saved in `resources/figures/`
4. User behavior patterns will be saved in `user_habits_analysis.txt`

## Authors

Created for the Formal Methods for Computer Science course at UniBA

## Dataset Citation and Acknowledgments

This work uses the CASAS TM001 Single-resident home dataset from Washington State University. If you use this dataset in your work, please cite:

Special thanks to the CASAS team at Washington State University for providing this valuable dataset for research purposes. The dataset is available at: https://casas.wsu.edu/datasets/