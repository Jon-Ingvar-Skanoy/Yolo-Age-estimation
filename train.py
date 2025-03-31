import argparse
import os
from ultralytics import YOLO
import yaml
import torch
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
    
    Args:
        data_yaml: Path to the data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        img_size: Input image size
        batch_size: Batch size for training
        epochs: Number of epochs to train
        workers: Number of worker threads for data loading
        device: Device to train on ('cpu', '0', '0,1', etc.)
        output_dir: Directory to save training results
        pretrained: Whether to use pretrained weights
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

def validate_model(model, data_yaml, img_size=640, batch_size=16, device='0'):
    """
    Validate a trained YOLOv8 model on the validation set.
    
    Args:
        model: Trained YOLO model
        data_yaml: Path to the data.yaml file
        img_size: Input image size
        batch_size: Batch size for validation
        device: Device to validate on
    """
    # Run validation
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=True
    )
    
    # Print validation results
    print("\nValidation Results:")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"Precision: {results.box.p:.4f}")
    print(f"Recall: {results.box.r:.4f}")
    
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
    parser.add_argument('--epochs', type=int, default=100,
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
    
    # Validate model
    validate_model(
        model=model,
        data_yaml=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )