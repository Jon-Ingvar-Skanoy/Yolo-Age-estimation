import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import yaml
from datetime import datetime

def train_yolov8(
    data_yaml,
    model_size='n',
    img_size=640,
    batch_size=16,
    epochs=100,
    workers=4,
    device='0',
    output_dir='runs/train',
    pretrained=True
):
    """
    Train a YOLOv8 model on a custom dataset.
    """
    # Load the dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Display dataset information
    print(f"Training on dataset with {data_config['nc']} classes:")
    for idx, class_name in data_config['names'].items():
        print(f"  Class {idx}: {class_name}")
    
    # Set up the model
    if pretrained:
        model = YOLO(f'yolov8{model_size}.pt')
    else:
        model = YOLO(f'yolov8{model_size}.yaml')
    
    # Create timestamp for run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"age_detection_{timestamp}"
    
    # Train the model
    results = model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        workers=workers,
        device=device,
        project=output_dir,
        name=run_name,
        verbose=True,
        patience=50,  # Early stopping patience
        save=True,  # Save best model
        val=True    # Validate during training
    )
    
    # Print training summary
    print("Training complete!")
    print(f"Best mAP50-95: {results.fitness:.4f}")
    print(f"Results saved to {os.path.join(output_dir, run_name)}")
    
    return model, results

def validate_model_with_confusion_matrix(model, data_yaml, img_size=640, batch_size=16, device='0'):
    """
    Validate a trained YOLOv8 model and generate a confusion matrix.
    """
    # Load class names
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        
    class_names = []
    # Replace your class names loading code with this
    class_names = []
    if 'names' in data_config:
        # If names is a dictionary
        if isinstance(data_config['names'], dict):
            num_classes = data_config.get('nc', len(data_config['names']))
            class_names = [""] * num_classes
            
            for key, value in data_config['names'].items():
                # Convert key to int if it's a string that represents an integer
                idx = key  # This works with integer keys directly
                if isinstance(key, str) and key.isdigit():
                    idx = int(key)
                    
                if isinstance(idx, int) and 0 <= idx < num_classes:
                    class_names[idx] = value
        # If names is a list
        elif isinstance(data_config['names'], list):
            class_names = data_config['names']
    
    # Run validation
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=True
    )
    
    # Print validation results safely
    print("\nValidation Results:")
    try:
        map50_95 = float(results.box.map)
        map50 = float(results.box.map50)
        precision = float(results.box.p)
        recall = float(results.box.r)
        
        print(f"mAP50-95: {map50_95:.4f}")
        print(f"mAP50: {map50:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    except Exception as e:
        print(f"Could not print detailed metrics: {e}")
        print("Basic results:", results)
    
    # Generate confusion matrix
    try:
        # Get predictions and ground truth
        conf_matrix = results.confusion_matrix.matrix
        
        # Normalize to get percentages
        conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')
        
        # Save the confusion matrix
        cm_path = os.path.join(os.path.dirname(results.save_dir), 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"Failed to generate confusion matrix: {e}")
        
        # Alternative approach if the built-in confusion matrix isn't available
        print("Attempting alternative confusion matrix calculation...")
        try:
            # Get all predictions and ground truths
            y_true = []
            y_pred = []
            
            # Run inference on validation set to get predictions
            val_folder = os.path.join(os.path.dirname(data_yaml), data_config['val'])
            
            # This approach requires manually iterating through the validation dataset
            # and collecting predictions, which is beyond the scope of this fix
            
            print("Manual confusion matrix calculation requires additional implementation.")
        except Exception as e2:
            print(f"Alternative confusion matrix calculation failed: {e2}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 on age detection dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x) (default: n)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for data loading (default: 4)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on (cpu, 0, 0,1, etc.) (default: 0)')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                        help='Directory to save training results (default: runs/train)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch instead of using pretrained weights')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Train model
    model, results = train_yolov8(
        data_yaml=args.data,
        model_size=args.model_size,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
        device=args.device,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained
    )
    
    # Validate model with confusion matrix
    validate_model_with_confusion_matrix(
        model=model,
        data_yaml=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )