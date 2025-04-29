import argparse
import os
import torch
import yaml
import json
import optuna
from ultralytics import YOLO
from datetime import datetime

def objective(trial, model_path, data_yaml, epochs=10, device='0'):
    """
    Optuna objective function for YOLO hyperparameter tuning.
    """
    # Define the hyperparameters to tune
    params = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),
        'lrf': trial.suggest_float('lrf', 0.01, 1.0),
        'momentum': trial.suggest_float('momentum', 0.6, 0.95),
        'weight_decay': trial.suggest_float('weight_decay', 0.0001, 0.001, log=True),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
        'warmup_momentum': trial.suggest_float('warmup_momentum', 0.5, 0.95),
        'box': trial.suggest_float('box', 0.02, 0.2),
        'cls': trial.suggest_float('cls', 0.2, 4.0),
        'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.1),
        'hsv_s': trial.suggest_float('hsv_s', 0.5, 0.9),
        'hsv_v': trial.suggest_float('hsv_v', 0.5, 0.9),
        'degrees': trial.suggest_float('degrees', 0.0, 45.0),
        'translate': trial.suggest_float('translate', 0.0, 0.5),
        'scale': trial.suggest_float('scale', 0.0, 0.5),
        'fliplr': trial.suggest_float('fliplr', 0.0, 0.5),
        'batch': trial.suggest_categorical('batch', [8, 16, 32]),
        'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
        'imgsz': 416,
        'optimizer': 'AdamW',
        'val': False
    }
    
    # Create a new model instance for each trial to avoid state leakage
    model = YOLO(model_path)
    
    # Generate a unique run name for this trial
    trial_name = f"trial_{trial.number}"
    
    # Train with the suggested hyperparameters
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            device=device,
            verbose=False,
            plots=False,  # Disable plots for faster training
            **params
        )
        
        # Extract validation metrics (mAP50-95)
        if hasattr(results, 'fitness'):
            map50_95 = float(results.fitness)
        else:
            # Fallback to a default low value if results don't contain fitness
            map50_95 = 0.0
        
        return map50_95
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a very low score to discourage this parameter combination
        return 0.0

def run_optuna_tuning(
    data_yaml,
    model_size='n',
    output_dir='runs/tune_optuna',
    n_trials=40,
    epochs_per_trial=50,
    device='0'
):
    """
    Run hyperparameter tuning using Optuna.
    """
    # Load the dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Display dataset information
    print(f"Setting up Optuna tuning for dataset with {data_config.get('nc', 0)} classes:")
    for idx, class_name in data_config['names'].items():
        print(f"  Class {idx}: {class_name}")
    
    # Create timestamp for run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"age_detection_optuna_{timestamp}"
    
    # Create output directory
    output_path = os.path.join(output_dir, run_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Define the model path
    model_path = f'yolov8{model_size}.pt'
    
    # Create and configure the study
    study = optuna.create_study(direction='maximize', study_name="3 model size " + model_size, storage="sqlite:///YOLO.db", load_if_exists=True)
    
    # Run the optimization
    print(f"Starting Optuna hyperparameter tuning with {n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, model_path, data_yaml, epochs=epochs_per_trial, device=device),
        n_trials=n_trials
    )
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Save the best parameters to a JSON file
    best_params_path = os.path.join(output_path, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save a summary of the tuning process
    summary_path = os.path.join(output_path, 'tuning_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Optuna Hyperparameter Tuning Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {data_yaml}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of trials: {n_trials}\n")
        f.write(f"Epochs per trial: {epochs_per_trial}\n\n")
        f.write(f"Best Performance (mAP50-95): {best_value:.4f}\n\n")
        f.write(f"Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\nTop 10 Trials:\n")
        best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:10]
        for i, trial in enumerate(best_trials):
            f.write(f"\nRank {i+1} (Trial {trial.number}):\n")
            f.write(f"  Value: {trial.value:.4f}\n")
            f.write(f"  Params:\n")
            for param, value in trial.params.items():
                f.write(f"    {param}: {value}\n")
    
    # Save the full study as well
    study_path = os.path.join(output_path, 'study.pkl')
    optuna.study.store_study(study, study_path, pickle_study=True)
    
    print(f"Best parameters (mAP50-95: {best_value:.4f}):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"Best parameters saved to {best_params_path}")
    print(f"Tuning summary saved to {summary_path}")
    
    # Train a final model with the best parameters
    print("\nTraining final model with best parameters...")
    final_model = YOLO(model_path)
    final_results = final_model.train(
        data=data_yaml,
        epochs=epochs_per_trial * 5,  # Train for 5x longer with the best params
        device=device,
        project=output_dir,
        name=f"{run_name}_final",
        **best_params
    )
    
    print(f"Final model trained and saved to {output_dir}/{run_name}_final")
    
    return best_params, best_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna-based Hyperparameter Tuning for YOLOv8')
    parser.add_argument('--data', type=str, default= "data/age_dataset_test2/data.yaml" ,
                        help='Path to data.yaml file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x) (default: n)')
    parser.add_argument('--n-trials', type=int, default=40,
                        help='Number of Optuna trials (default: 40)')
    parser.add_argument('--epochs-per-trial', type=int, default=30,
                        help='Number of epochs to train in each trial (default: 10)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on (cpu, 0, 0,1, etc.) (default: 0)')
    parser.add_argument('--output-dir', type=str, default='runs/tune_optuna',
                        help='Directory to save tuning results (default: runs/tune_optuna)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Install optuna if not already installed
    try:
        import optuna
    except ImportError:
        print("Optuna not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "optuna"])
        import optuna
    
    # Run the Optuna tuning
    best_params, best_value = run_optuna_tuning(
        data_yaml=args.data,
        model_size=args.model_size,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        device=args.device
    )