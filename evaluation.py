import yaml
import os
import itertools
import copy
from pathlib import Path
import sys
import importlib
import traceback

from main import run_augmentation
from classifiers.classifier_small import main_training_function

# Define parameter sweep configuration
parameter_sweep = {
    'mode': 'individual',  # 'individual' or 'combination'
    'concatenate': False,
    'parameters': {
        # Parameters to vary in format 'section.parameter': [values]
        # 'output.positive_overlay_range': [[0, 1], [0, 2], [0, 3]],
        # 'output.negative_overlay_range': [[0, 0], [0, 1], [0, 2]],
        # 'output.snr_range': [[0.3, 1], [0.5, 1], [0.7, 1], [0.9, 1]],
        # 'output.repetitions': [[1, 1], [1, 2], [1, 3]],
        # 'output.n': [6000,7000,8000,9000,10000]
        # 'output.n': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] #Figure 2 left
        'input.limit_positives': [2, 3, 5, 10, 15, 20, 25, 30], #Figure 2 right
    },
    # Base directory for outputs
    'project_dir': 'classifiers',
    'results_dir': 'evaluation_results'
}

# Create results directory
results_dir = Path(parameter_sweep['project_dir']) / parameter_sweep['results_dir']
results_dir.mkdir(parents=True, exist_ok=True)

# Function to update nested dictionary values using dot notation
def update_nested_dict(d, key, value):
    keys = key.split('.')
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value
    return d

# Safely import the augmentation function
def get_augmentation_function():
    # We'll import the function dynamically to avoid issues
    # with code executing during import
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    from main import run_augmentation
    return run_augmentation

# Safely import the training function
def get_training_function():
    # Similarly import the training function dynamically
    import sys
    sys.path.insert(0, os.path.abspath('./classifiers'))
    from classifiers.classifier_small import main_training_function
    return main_training_function

# Function to run a single experiment pipeline
def run_experiment(experiment_id, config, use_case, parameters_desc):
    """Run a single experiment with the given parameters"""
    print(f"\n=== Experiment {experiment_id}: {parameters_desc} ===")

    # Save configs for reference
    augmentation_config_path = f'{results_dir}/augmentation_config_{experiment_id}.yaml'
    training_config_path = f'{results_dir}/training_config_{experiment_id}.yaml'
    
    with open(augmentation_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Update use_case with experiment ID
    use_case['experiment'] = experiment_id
    use_case['project_root'] = parameter_sweep['project_dir']
    use_case['experiment_dir'] = parameter_sweep['results_dir']
    with open(training_config_path, 'w') as f:
        yaml.dump(use_case, f)
    
    # Step 1: Run data augmentation in a controlled way
    try:
        print(f"Generating augmented dataset...")
        augmentation_fn = get_augmentation_function()
        augmentation_fn(config)
        print(f"Data augmentation completed successfully.")
    except Exception as e:
        print(f"Error during data augmentation: {str(e)}")
        traceback.print_exc()
        return False
    
    # Step 2: Run model training in a controlled way
    try:
        print(f"Running model training for experiment {experiment_id}...")
        training_fn = get_training_function()
        metrics = training_fn(use_case)
        
        # Save metrics
        with open(f'{results_dir}/metrics_{experiment_id}.yaml', 'w') as f:
            yaml.dump(metrics, f)
        
        print(f"Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        return False
    
    # Save experiment parameters
    with open(f'{results_dir}/experiment_{experiment_id}_parameters.yaml', 'w') as f:
        yaml.dump({
            'parameters': parameters_desc, 
            'experiment_id': experiment_id,
            'status': 'completed'
        }, f)
    
    print(f"=== Completed Experiment {experiment_id} ===\n")
    return True

# Main execution
def run_parameter_sweep():
    # Load the default configs
    with open('config.yaml', 'r') as f:
        default_config = yaml.safe_load(f)

    with open('classifiers/use_case.yaml', 'r') as f:
        default_use_case = yaml.safe_load(f)

    experiment_id = 1000
    # check if theres a previous experiment with Exp_ anything
    max_experiment_id = 0
    other_experiments_dir = parameter_sweep['project_dir'] + '/' + parameter_sweep['results_dir']
    if os.path.exists(other_experiments_dir):
        # get all the directories in the results directory
        dirs = [f for f in os.listdir(other_experiments_dir) if os.path.isdir(os.path.join(other_experiments_dir, f))]
        # get the last experiment id
        for dir_path in dirs:
            if dir_path.startswith('Exp_'):
                max_experiment_id = max(max_experiment_id, int(dir_path.split('_')[1]))
    if experiment_id <= max_experiment_id:
        raise ValueError(f"Experiment ID {experiment_id} is below max existing ID. Please choose a higher ID.")

    completed_experiments = []
    failed_experiments = []

    if parameter_sweep['mode'] == 'individual':
        # Option 1: Run one session for each parameter value, keeping others default
        for param_key, param_values in parameter_sweep['parameters'].items():
            for value in param_values:
                config = copy.deepcopy(default_config)
                if parameter_sweep['concatenate']:
                    config['output']['concatenate'] = True
                use_case = copy.deepcopy(default_use_case)
                update_nested_dict(config, param_key, value)
                
                parameters_desc = {param_key: value}
                success = run_experiment(experiment_id, config, use_case, parameters_desc)
                
                if success:
                    completed_experiments.append(experiment_id)
                else:
                    failed_experiments.append(experiment_id)
                
                experiment_id += 1
                
    elif parameter_sweep['mode'] == 'combination':
        # Option 2: Run one session for each combination of parameters
        param_keys = list(parameter_sweep['parameters'].keys())
        param_values = list(parameter_sweep['parameters'].values())
        
        for combination in itertools.product(*param_values):
            config = copy.deepcopy(default_config)
            if parameter_sweep['concatenate']:
                config['output']['concatenate'] = True
            use_case = copy.deepcopy(default_use_case)
            parameters_desc = {}
            
            for i, key in enumerate(param_keys):
                update_nested_dict(config, key, combination[i])
                parameters_desc[key] = combination[i]
            
            success = run_experiment(experiment_id, config, use_case, parameters_desc)
            
            if success:
                completed_experiments.append(experiment_id)
            else:
                failed_experiments.append(experiment_id)
            
            experiment_id += 1
    else:
        print("Invalid mode. Please choose 'individual' or 'combination'")

    # Summarize results
    print(f"\n=== Parameter Sweep Summary ===")
    print(f"Total experiments: {experiment_id-1}")
    print(f"Completed experiments: {len(completed_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"Failed experiment IDs: {failed_experiments}")

if __name__ == "__main__":
    run_parameter_sweep()