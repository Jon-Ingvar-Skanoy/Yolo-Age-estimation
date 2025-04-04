import argparse
import os
import json
import torch
import yaml
from ultralytics import YOLO
from datetime import datetime
import optuna


def train_with_best_params(
    data_yaml,
    model_size='n',
    study_path=None,
    study_name=None,
    db_path=None,
    epochs=100,
    device='0',
    project='runs/train_best_params',
    name=None
):
    """
    Train a YOLO model with the best hyperparameters from an Optuna study.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        study_path: Path to saved study.pkl file
        study_name: Name of the study in the database
        db_path: Path to Optuna database (sqlite:///YOLO.db)
        epochs: Number of epochs to train
        device: Device to run on (cpu, 0, 0,1, etc.)
        project: Directory to save training results
        name: Name for the training run
    """
    # Load the dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Display dataset information
    print(f"Dataset information:")
    print(f"  Classes: {data_config.get('nc', 0)}")
    for idx, class_name in enumerate(data_config['names']):
        print(f"  Class {idx}: {class_name}")
    
    # Create timestamp for run directory if name not provided
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"best_params_{timestamp}"
    
    # Create output directory
    os.makedirs(project, exist_ok=True)
    
    # Define the model path
    model_path = f'yolov8{model_size}.pt'
    print(f"Using model: {model_path}")
    
    # Load the best hyperparameters from the study
    if study_path and os.path.exists(study_path):
        print(f"Loading Optuna study from file: {study_path}")
        study = optuna.load_study(study_name="loaded_study", storage=study_path)
    elif db_path and study_name:
        print(f"Loading Optuna study from database: {db_path}, study name: {study_name}")
        study = optuna.load_study(study_name=study_name, storage=db_path)
    else:
        raise ValueError("Either study_path or (db_path and study_name) must be provided")
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Print the best parameters
    print(f"Best parameters (mAP50-95: {best_value:.4f}):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save the best parameters to a JSON file
    params_dir = os.path.join(project, name)
    os.makedirs(params_dir, exist_ok=True)
    best_params_path = os.path.join(params_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Best parameters saved to {best_params_path}")
    
    # Create a new model instance
    model = YOLO(model_path)
    
    # Add fixed parameters that might not be in the study
    training_params = best_params.copy()
    # Add imgsz if not in best_params
    if 'imgsz' not in training_params:
        training_params['imgsz'] = 416

    
    # add optimizer 
    training_params['optimizer'] = 'AdamW'
    
    # Train with the best parameters
    print(f"\nStarting training with best parameters for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        **training_params
    )
    
    # Print training results
    print(f"\nTraining completed.")
    print(f"Results saved to {os.path.join(project, name)}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 with best hyperparameters from Optuna study')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x) (default: n)')
    
    # Optuna study loading options
    study_group = parser.add_mutually_exclusive_group(required=True)
    study_group.add_argument('--study-path', type=str,
                        help='Path to saved study.pkl file')
    study_group.add_argument('--db-path', type=str,
                        help='Path to Optuna database (e.g., sqlite:///YOLO.db)')
    
    parser.add_argument('--study-name', type=str,
                        help='Name of the study in the database (required with --db-path)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on (cpu, 0, 0,1, etc.) (default: 0)')
    parser.add_argument('--project', type=str, default='runs/train_best_params',
                        help='Directory to save training results (default: runs/train_best_params)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the training run (default: timestamp-based name)')
    
    args = parser.parse_args()
    
    # Check if study-name is provided when using db-path
    if args.db_path and not args.study_name:
        parser.error("--study-name is required when using --db-path")
    
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run training with best parameters
    results = train_with_best_params(
        data_yaml=args.data,
        model_size=args.model_size,
        study_path=args.study_path,
        study_name=args.study_name,
        db_path=args.db_path,
        epochs=args.epochs,
        device=args.device,
        project=args.project,
        name=args.name
    )