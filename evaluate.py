import argparse
import os
import json
import yaml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model accuracy with best parameters')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained YOLO model (.pt file)')
    parser.add_argument('--params-file', type=str, required=True,
                        help='Path to the best_params.json file from Optuna tuning')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for predictions (default: 0.25)')
    parser.add_argument('--output-dir', type=str, default='results/accuracy',
                        help='Directory to save evaluation results (default: results/accuracy)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the best parameters
    with open(args.params_file, 'r') as f:
        best_params = json.load(f)
    
    print(f"Loaded best parameters from {args.params_file}")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Load the model
    model = YOLO(args.model_path)
    
    print(f"Evaluating model accuracy on {args.data}")
    accuracy, results = calculate_accuracy(model, args.data, args.conf_threshold)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'accuracy_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f"\nAccuracy Results:")
    print(f"Overall Accuracy: {accuracy:.4f} ({results['correct_predictions']}/{results['total_images']})")
    print("\nPer-class Accuracy:")
    
    # Load class names
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    
    # Print per-class results
    for class_id, stats in results['per_class'].items():
        if stats['total'] > 0:
            print(f"  {class_names[int(class_id)]}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print(f"\nDetailed results saved to {results_path}")


if __name__ == "__main__":
    main()