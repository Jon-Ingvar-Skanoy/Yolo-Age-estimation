import argparse
import os
import json
import yaml
import torch
import numpy as np
import random
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime
import optuna
from pathlib import Path
import shutil
import pandas as pd


def get_best_params(study_path=None, study_name=None, db_path=None):
    """
    Get the best hyperparameters from an Optuna study.
    
    Args:
        study_path: Path to saved study.pkl file
        study_name: Name of the study in the database
        db_path: Path to Optuna database (sqlite:///YOLO.db)
        
    Returns:
        best_params: Dictionary of best parameters
        best_value: Best value (mAP50-95) from the study
    """
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
    
    return best_params, best_value


def train_model(data_yaml, model_size, best_params, run_id, epochs=100, device='0', 
                project='runs/multi_runs', base_name=None):
    """
    Train a YOLO model with the given parameters.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        best_params: Dictionary of hyperparameters
        run_id: ID of the current run (for naming)
        epochs: Number of epochs to train
        device: Device to run on (cpu, 0, 0,1, etc.)
        project: Directory to save training results
        base_name: Base name for the training run
        
    Returns:
        results: Training results
        run_dir: Directory where results are saved
    """
    # Set different random seeds for each run to ensure stochasticity
    random_seed = random.randint(0, 10000)  # Generate a random seed for this run
    print(f"Run {run_id}: Using random seed {random_seed}")
    
    # Set seeds for different libraries
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        # Disable deterministic operations for better performance and true randomness
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Create a timestamp-based name if not provided
    if base_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"best_params_{timestamp}"
    
    # Create a name for this specific run
    name = f"{base_name}_run{run_id}"
    
    # Define the model path
    model_path = f'yolov8{model_size}.pt'
    print(f"Run {run_id}: Using model {model_path}")
    
    # Create a new model instance
    model = YOLO(model_path)
    
    # Add fixed parameters that might not be in the study
    training_params = best_params.copy()
    # Add imgsz if not in best_params
    if 'imgsz' not in training_params:
        training_params['imgsz'] = 416
    
    # Add the random seed to the training parameters
    training_params['seed'] = random_seed
    training_params['optimizer'] = "AdamW"
    
    # Train with the best parameters
    print(f"\nRun {run_id}: Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        **training_params
    )
    
    run_dir = os.path.join(project, name)
    print(f"Run {run_id}: Training completed. Results saved to {run_dir}")
    
    return results, run_dir


def calculate_accuracy(model, data_yaml, conf_threshold=0.25):
    """
    Calculate accuracy score for each image by comparing the most confident
    detected object with the ground truth label.
    
    Args:
        model: YOLO model
        data_yaml: Path to data.yaml file
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        accuracy: Accuracy score
        results_dict: Dictionary with detailed results
    """
    # Load dataset information
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get the class names
    class_names = data_config['names']
    
    # Get validation dataset path
    val_path = data_config.get('val')
    if not val_path:
        raise ValueError("Validation set path not found in data.yaml")
    
    # If val_path is relative, make it absolute based on the data.yaml location
    data_dir = os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(val_path):
        val_path = os.path.join(data_dir, val_path)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    # Handle if val_path is a file with paths
    if os.path.isfile(val_path) and val_path.endswith('.txt'):
        with open(val_path, 'r') as f:
            for line in f:
                img_path = line.strip()
                # Convert relative paths to absolute if needed
                if not os.path.isabs(img_path):
                    img_path = os.path.join(data_dir, img_path)
                if os.path.exists(img_path):
                    image_files.append(img_path)
    # Handle if val_path is a directory
    elif os.path.isdir(val_path):
        for root, _, files in os.walk(val_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        raise ValueError(f"Invalid validation path: {val_path}")
    
    if not image_files:
        raise ValueError(f"No images found in validation set path: {val_path}")
    
    print(f"Found {len(image_files)} images in the validation set")
    
    # Initialize counters
    correct_predictions = 0
    total_images = 0
    results_dict = {
        "per_image": [],
        "per_class": {class_id: {"correct": 0, "total": 0} for class_id in class_names}
    }
    
    # Process each image
    for img_path in tqdm(image_files, desc="Evaluating images"):
        # Get corresponding label file
        label_path = get_label_path(img_path)
        
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_path}")
            continue
        
        # Read ground truth labels
        ground_truth_classes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id x y w h
                    class_id = int(float(parts[0]))
                    ground_truth_classes.append(class_id)
        
        if not ground_truth_classes:
            print(f"Warning: No valid labels in {label_path}")
            continue
        
        # Run inference
        results = model(img_path, conf=conf_threshold)[0]
        
        # Get predictions
        predictions = results.boxes.data.cpu().numpy()
        
        # Sort predictions by confidence (descending)
        if len(predictions) > 0:
            # Sort by confidence (5th column, index 4)
            predictions = predictions[predictions[:, 4].argsort()[::-1]]
            
            # Get the most confident prediction
            most_confident_pred = predictions[0]
            pred_class_id = int(most_confident_pred[5])
            
            # Check if prediction matches any ground truth
            is_correct = pred_class_id in ground_truth_classes
            
            if is_correct:
                correct_predictions += 1
            
            # Update per-class statistics
            for gt_class in set(ground_truth_classes):  # Count each class only once per image
                results_dict["per_class"][gt_class]["total"] += 1
                if is_correct and pred_class_id == gt_class:
                    results_dict["per_class"][gt_class]["correct"] += 1
            
            # Store per-image results
            results_dict["per_image"].append({
                "image_path": img_path,
                "ground_truth": [class_names[cls] for cls in ground_truth_classes],
                "prediction": class_names[pred_class_id],
                "confidence": float(most_confident_pred[4]),
                "correct": is_correct
            })
        else:
            # No detections
            results_dict["per_image"].append({
                "image_path": img_path,
                "ground_truth": [class_names[cls] for cls in ground_truth_classes],
                "prediction": "none",
                "confidence": 0.0,
                "correct": False
            })
            
            # Update per-class statistics (all are incorrect since no detection)
            for gt_class in set(ground_truth_classes):
                results_dict["per_class"][gt_class]["total"] += 1
        
        total_images += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    # Calculate per-class accuracy
    for class_id in results_dict["per_class"]:
        class_total = results_dict["per_class"][class_id]["total"]
        class_correct = results_dict["per_class"][class_id]["correct"]
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        results_dict["per_class"][class_id]["accuracy"] = class_accuracy
    
    results_dict["overall_accuracy"] = accuracy
    results_dict["total_images"] = total_images
    results_dict["correct_predictions"] = correct_predictions
    
    return accuracy, results_dict


def get_label_path(img_path):
    """
    Convert image path to label path for YOLO format datasets.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        label_path: Path to the corresponding label file
    """
    img_path = Path(img_path)
    
    # Try common YOLO folder structures:
    # 1. Standard YOLOv5/YOLOv8 structure: 'images/train' -> 'labels/train'
    if 'images' in img_path.parts:
        idx = img_path.parts.index('images')
        label_path = Path(*img_path.parts[:idx], 'labels', *img_path.parts[idx+1:])
        label_path = label_path.with_suffix('.txt')
        if label_path.exists():
            return str(label_path)
    
    # 2. Replace image extension with .txt in the same directory
    label_path = img_path.with_suffix('.txt')
    if label_path.exists():
        return str(label_path)
    
    # 3. Check for a 'labels' directory at the same level as the image directory
    parent_dir = img_path.parent
    label_dir = parent_dir.parent / 'labels' / parent_dir.name
    label_path = label_dir / img_path.name
    label_path = label_path.with_suffix('.txt')
    if label_path.exists():
        return str(label_path)
    
    # 4. Default fallback: just replace extension
    return str(img_path.with_suffix('.txt'))


def read_metrics_from_run(run_dir):
    """
    Read validation metrics from a training run directory.
    
    Args:
        run_dir: Path to the training run directory
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics_file = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found at {metrics_file}")
        return {}
    
    # Read the CSV file and get the last row (final epoch metrics)
    try:
        df = pd.read_csv(metrics_file)
        final_metrics = df.iloc[-1].to_dict()
        return final_metrics
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        return {}


def run_multiple_trainings(
    data_yaml,
    model_size='n',
    study_path=None,
    study_name=None,
    db_path=None,
    num_runs=5,
    epochs=100,
    device='0',
    project='runs/multi_runs',
    conf_threshold=0.25
):
    """
    Run multiple training runs with the best hyperparameters and calculate average metrics.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        study_path: Path to saved study.pkl file
        study_name: Name of the study in the database
        db_path: Path to Optuna database (sqlite:///YOLO.db)
        num_runs: Number of training runs
        epochs: Number of epochs per run
        device: Device to run on (cpu, 0, 0,1, etc.)
        project: Directory to save training results
        conf_threshold: Confidence threshold for accuracy calculation
    """
    # Load dataset information
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Display dataset information
    print(f"Dataset information:")
    print(f"  Classes: {data_config.get('nc', 0)}")
    for idx, class_name in enumerate(data_config['names']):
        print(f"  Class {idx}: {class_name}")
    
    # Get the best hyperparameters
    best_params, best_value = get_best_params(study_path, study_name, db_path)
    
    # Print the best parameters
    print(f"Best parameters (mAP50-95: {best_value:.4f}):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Create timestamp for run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"multi_run_{timestamp}"
    
    # Create project directory
    os.makedirs(project, exist_ok=True)
    
    # Create directory for aggregate results
    aggregate_dir = os.path.join(project, f"{base_name}_aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Save the best parameters to a JSON file in the aggregate directory
    best_params_path = os.path.join(aggregate_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Track metrics across runs
    all_metrics = []
    all_accuracies = []
    
    # Run multiple training iterations
    for run_id in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"Starting Run {run_id}/{num_runs}")
        print(f"{'='*80}")
        
        # Train model with best parameters
        results, run_dir = train_model(
            data_yaml=data_yaml,
            model_size=model_size,
            best_params=best_params,
            run_id=run_id,
            epochs=epochs,
            device=device,
            project=project,
            base_name=base_name
        )
        
        # Get the path to the best weights
        weights_path = os.path.join(run_dir, 'weights', 'best.pt')
        if not os.path.exists(weights_path):
            print(f"Warning: Best weights not found at {weights_path}")
            continue
        
        # Load the trained model for evaluation
        print(f"\nRun {run_id}: Loading model from {weights_path} for evaluation")
        trained_model = YOLO(weights_path)
        
        # Calculate accuracy
        print(f"Run {run_id}: Calculating accuracy metrics")
        accuracy, accuracy_results = calculate_accuracy(trained_model, data_yaml, conf_threshold)
        
        # Read standard YOLO metrics from results.csv
        yolo_metrics = read_metrics_from_run(run_dir)
        
        # Combine metrics
        combined_metrics = {
            'run_id': run_id,
            'accuracy': accuracy,
            **yolo_metrics
        }
        all_metrics.append(combined_metrics)
        all_accuracies.append(accuracy_results)
        
        # Save accuracy results to JSON
        accuracy_results_path = os.path.join(run_dir, 'accuracy_results.json')
        with open(accuracy_results_path, 'w') as f:
            json.dump(accuracy_results, f, indent=4)
        
        # Also make a copy in the aggregate directory
        shutil.copy(accuracy_results_path, os.path.join(aggregate_dir, f'accuracy_results_run{run_id}.json'))
        
        # Print metrics summary for this run
        print(f"\nRun {run_id} Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy_results['correct_predictions']}/{accuracy_results['total_images']})")
        print("  YOLO Metrics:")
        for key, value in yolo_metrics.items():
            if key.startswith('metrics/'):
                print(f"    {key.replace('metrics/', '')}: {value:.4f}")
    
    # Calculate average metrics
    if all_metrics:
        # Create a DataFrame for easier analysis
        metrics_df = pd.DataFrame(all_metrics)
        
        # Calculate mean and standard deviation
        mean_metrics = metrics_df.mean(numeric_only=True)
        std_metrics = metrics_df.std(numeric_only=True)
        
        # Save all metrics to CSV
        metrics_df.to_csv(os.path.join(aggregate_dir, 'all_runs_metrics.csv'), index=False)
        
        # Save average metrics to CSV
        avg_metrics_df = pd.DataFrame({
            'metric': mean_metrics.index,
            'mean': mean_metrics.values,
            'std': std_metrics.values
        })
        avg_metrics_df.to_csv(os.path.join(aggregate_dir, 'average_metrics.csv'), index=False)
        
        # Print individual values for key metrics to verify stochasticity
        print("\nIndividual run values for key metrics:")
        for metric in ['accuracy', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].tolist()
                print(f"  {metric}: {values}")
        
        # Also save in a more readable format
        summary_path = os.path.join(aggregate_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Multiple Training Runs Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {data_yaml}\n")
            f.write(f"Model: yolov8{model_size}.pt\n")
            f.write(f"Number of runs: {num_runs}\n")
            f.write(f"Epochs per run: {epochs}\n\n")
            
            f.write(f"Average Metrics (mean +/- std):\n")
            for index, row in avg_metrics_df.iterrows():
                metric = row['metric']
                mean = row['mean']
                std = row['std']
                f.write(f"  {metric}: {mean:.4f} +/- {std:.4f}\n")
            
            f.write("\nIndividual Run Results:\n")
            for run_id in range(1, num_runs + 1):
                run_metrics = metrics_df[metrics_df['run_id'] == run_id]
                if not run_metrics.empty:
                    f.write(f"\nRun {run_id}:\n")
                    for column in run_metrics.columns:
                        if column != 'run_id':
                            value = run_metrics[column].values[0]
                            if isinstance(value, (int, float)):
                                f.write(f"  {column}: {value:.4f}\n")
                    
                    # Include the random seed used for this run
                    run_dir = os.path.join(project, f"{base_name}_run{run_id}")
                    seed_file = os.path.join(run_dir, 'args.yaml')
                    if os.path.exists(seed_file):
                        try:
                            with open(seed_file, 'r') as sf:
                                args = yaml.safe_load(sf)
                                if 'seed' in args:
                                    f.write(f"  random_seed: {args['seed']}\n")
                        except Exception as e:
                            f.write(f"  Error reading seed: {e}\n")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Average Results Across {num_runs} Runs:")
        print(f"{'='*80}")
        print(f"  Accuracy: {mean_metrics['accuracy']:.4f} +/- {std_metrics['accuracy']:.4f}")
        print("  YOLO Metrics:")
        for metric in mean_metrics.index:
            if metric.startswith('metrics/'):
                metric_name = metric.replace('metrics/', '')
                print(f"    {metric_name}: {mean_metrics[metric]:.4f} +/- {std_metrics[metric]:.4f}")
        
        print(f"\nDetailed results saved to {aggregate_dir}")
    else:
        print("\nNo valid runs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple YOLOv8 training runs with best parameters')
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
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of training runs (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs per run (default: 100)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on (cpu, 0, 0,1, etc.) (default: 0)')
    parser.add_argument('--project', type=str, default='runs/multi_runs',
                        help='Directory to save training results (default: runs/multi_runs)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for accuracy calculation (default: 0.25)')
    
    args = parser.parse_args()
    
    # Check if study-name is provided when using db-path
    if args.db_path and not args.study_name:
        parser.error("--study-name is required when using --db-path")
    
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run multiple training runs
    run_multiple_trainings(
        data_yaml=args.data,
        model_size=args.model_size,
        study_path=args.study_path,
        study_name=args.study_name,
        db_path=args.db_path,
        num_runs=args.num_runs,
        epochs=args.epochs,
        device=args.device,
        project=args.project,
        conf_threshold=args.conf_threshold
    )