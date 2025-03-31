import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class YOLODataset(Dataset):
    """Custom Dataset for YOLO training in PyTorch"""
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.img_size = img_size
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Transform image
        img_tensor = self.transform(img)
        
        # Get corresponding label file
        label_file = os.path.splitext(self.img_files[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        
        # Read labels (YOLO format: class x_center y_center width height)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = []
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        # Convert to [class_id, x_center, y_center, width, height] format
                        labels.append([class_id, x_center, y_center, width, height])
                
                if labels:
                    labels = torch.tensor(labels)
                else:
                    # Empty label
                    labels = torch.zeros((0, 5))
        else:
            # No label file
            labels = torch.zeros((0, 5))
            
        return img_tensor, labels, img_path

def load_yolo_for_pytorch_training(model_path=None, num_classes=1):
    """
    Load a YOLO model for training with PyTorch
    
    Args:
        model_path: Path to pretrained YOLO weights or model name (e.g., 'yolov8n.pt')
                   If None, will use a fresh yolov8n model
        num_classes: Number of classes to detect
        
    Returns:
        PyTorch model ready for training
    """
    # Use a pretrained model or start with a fresh model
    if model_path is None:
        model_path = 'yolov8n.pt'  # Use YOLOv8 nano as default
    
    # Load model directly as a PyTorch model
    model = DetectionModel(model_path)
    
    # If number of classes is different, modify the detection head
    if num_classes != model.nc:
        print(f"Modifying model to detect {num_classes} classes instead of {model.nc}")
        
        # Get the detection head (last layer)
        # In YOLOv8, for each detection layer, modify the output channels
        for m in model.model.modules():
            if hasattr(m, 'nc'):
                m.nc = num_classes
                # Adjust the prediction layer's output dimension
                if hasattr(m, 'no'):
                    m.no = num_classes + 5  # class + box (4) + objectness (1)
                    m.cv2.conv.out_channels = num_classes + 5
    
    # Set model to training mode
    model.train()
    
    return model

def train_yolo_with_pytorch(model, train_dataset, val_dataset=None, 
                           batch_size=16, epochs=100, lr=0.001, 
                           save_dir='yolo_pytorch_training'):
    """
    Train a YOLO model using PyTorch
    
    Args:
        model: PyTorch YOLO model
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save models and logs
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn  # Custom collate function to handle variable size labels
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=collate_fn
        )
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            # Move to device
            imgs = imgs.to(model.device)
            
            # Convert targets to model input format
            targets_list = []
            for i, target in enumerate(targets):
                if len(target) > 0:
                    # Add batch index column
                    batch_idx_col = torch.full((len(target), 1), i, device=model.device)
                    # Format: [batch_idx, class_id, x, y, w, h]
                    target_with_batch = torch.cat([batch_idx_col, target], dim=1).to(model.device)
                    targets_list.append(target_with_batch)
            
            if targets_list:
                targets_tensor = torch.cat(targets_list, dim=0)
            else:
                targets_tensor = torch.zeros((0, 6), device=model.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_items = model(imgs, targets_tensor)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")
        
        # Validation phase
        if val_dataset:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_idx, (imgs, targets, _) in enumerate(val_loader):
                    # Move to device
                    imgs = imgs.to(model.device)
                    
                    # Convert targets to model input format (same as above)
                    targets_list = []
                    for i, target in enumerate(targets):
                        if len(target) > 0:
                            batch_idx_col = torch.full((len(target), 1), i, device=model.device)
                            target_with_batch = torch.cat([batch_idx_col, target], dim=1).to(model.device)
                            targets_list.append(target_with_batch)
                    
                    if targets_list:
                        targets_tensor = torch.cat(targets_list, dim=0)
                    else:
                        targets_tensor = torch.zeros((0, 6), device=model.device)
                    
                    # Compute loss
                    loss, _ = model(imgs, targets_tensor)
                    
                    # Update validation loss
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            print(f"Epoch: {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_dataset else None,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    print(f"Training completed. Final model saved.")
    
    return model

def collate_fn(batch):
    """
    Custom collate function for the DataLoader
    Handles variable size labels
    """
    imgs, targets, paths = zip(*batch)
    return torch.stack(imgs), targets, paths

def example_usage():
    """Example of how to load a YOLO model and train it with PyTorch"""
    # 1. Load a pretrained YOLO model
    model_path = "yolov8n.pt"  # Use pretrained YOLOv8 nano model
    # Or specify your trained model path
    # model_path = "runs/detect/yolov8n_face_detector/weights/best.pt"
    
    # Number of classes in your dataset
    num_classes = 1  # Just face detection
    
    # Load the model for PyTorch training
    model = load_yolo_for_pytorch_training(model_path, num_classes)
    
    # 2. Create datasets
    train_dataset = YOLODataset(
        img_dir="yolo_dataset/train/images",
        label_dir="yolo_dataset/train/labels",
        img_size=640
    )
    
    val_dataset = YOLODataset(
        img_dir="yolo_dataset/val/images",
        label_dir="yolo_dataset/val/labels",
        img_size=640
    )
    
    # 3. Train the model with PyTorch
    trained_model = train_yolo_with_pytorch(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        epochs=50,
        lr=0.001,
        save_dir="yolo_pytorch_training_results"
    )
    
    print("Training with PyTorch completed successfully!")
    
    # 4. Convert back to YOLO model for inference if needed
    yolo_model = YOLO()
    yolo_model.model = trained_model
    
    # Now you can use the yolo_model for inference
    # results = yolo_model("path/to/test/image.jpg")
    
    return trained_model, yolo_model

if __name__ == "__main__":
    pytorch_model, yolo_model = example_usage()