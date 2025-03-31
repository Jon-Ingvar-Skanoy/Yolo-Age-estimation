import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from tqdm import tqdm

# Import your dataloader
from dataset import FacialAgeDataset, get_age_prediction_dataloaders

# YOLO model architecture for age detection
class YOLOAgeDetector(nn.Module):
    def __init__(self, num_classes=11):
        super(YOLOAgeDetector, self).__init__()
        
        # Number of classes (age categories)
        self.num_classes = num_classes
        
        # Feature extraction backbone (similar to Darknet)
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Detection head
        # Output: (num_classes + 5) where 5 is [objectness, x, y, w, h]
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, self.num_classes + 5, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        # Get batch size and input dimensions
        batch_size, _, height, width = x.shape
        
        # Extract features
        features = self.backbone(x)
        
        # Apply detection head
        output = self.detection_head(features)
        
        # Reshape for final prediction (B, C, H, W) -> (B, H, W, C)
        output = output.permute(0, 2, 3, 1)
        
        # In YOLO the output is a 3D tensor where each cell predicts one object
        # For simplicity, we'll use a single prediction per grid cell
        
        return output


# Loss function for YOLO detection
class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets, device='cpu'):
        """
        Calculate YOLO loss
        
        Args:
            predictions: Model predictions (B, H, W, num_classes+5)
            targets: List of target tensors [class_id, x_center, y_center, width, height]
            
        Returns:
            loss: The combined loss value
        """
        batch_size = predictions.size(0)
        grid_size = predictions.size(1)  # Assuming square grid
        
        # Compute losses
        obj_loss = 0.0
        noobj_loss = 0.0
        coord_loss = 0.0
        class_loss = 0.0
        
        for b in range(batch_size):
            target_boxes = targets[b]
            
            # Skip if no targets for this image
            if target_boxes.size(0) == 0:
                continue
                
            for t in range(target_boxes.size(0)):
                # Get target values
                class_id = target_boxes[t, 0].long()
                x_center = target_boxes[t, 1] * grid_size
                y_center = target_boxes[t, 2] * grid_size
                width = target_boxes[t, 3] * grid_size
                height = target_boxes[t, 4] * grid_size
                
                # Get grid cell coordinates
                grid_x = int(x_center)
                grid_y = int(y_center)
                
                # Constrain to valid grid indices
                grid_x = min(max(grid_x, 0), grid_size - 1)
                grid_y = min(max(grid_y, 0), grid_size - 1)
                
                # Get prediction at grid cell
                pred = predictions[b, grid_y, grid_x]
                
                # Objectness loss (Binary cross entropy)
                obj_loss += self.bce(
                    pred[0].unsqueeze(0),
                    torch.ones(1, device=device)
                )
                
                # Coordinate loss (MSE)
                # x, y predictions
                x_pred = torch.sigmoid(pred[1])
                y_pred = torch.sigmoid(pred[2])
                
                # x, y targets (relative to grid cell)
                x_target = x_center - grid_x
                y_target = y_center - grid_y
                
                coord_loss += self.lambda_coord * (
                    self.mse(x_pred.unsqueeze(0), torch.tensor([x_target], device=device)) +
                    self.mse(y_pred.unsqueeze(0), torch.tensor([y_target], device=device))
                )
                
                # Width, height loss (MSE)
                w_pred = pred[3]
                h_pred = pred[4]
                
                w_target = width
                h_target = height
                
                coord_loss += self.lambda_coord * (
                    self.mse(w_pred.unsqueeze(0), torch.tensor([w_target], device=device)) +
                    self.mse(h_pred.unsqueeze(0), torch.tensor([h_target], device=device))
                )
                
                # Class prediction loss (Binary cross entropy)
                class_pred = pred[5:]
                class_target = torch.zeros_like(class_pred)
                class_target[class_id] = 1.0
                
                class_loss += self.bce(class_pred, class_target)
            
            # No object loss (for all cells that don't contain objects)
            for y in range(grid_size):
                for x in range(grid_size):
                    # Skip cells with objects
                    has_obj = False
                    for t in range(target_boxes.size(0)):
                        target_x = int(target_boxes[t, 1] * grid_size)
                        target_y = int(target_boxes[t, 2] * grid_size)
                        if x == target_x and y == target_y:
                            has_obj = True
                            break
                    
                    if not has_obj:
                        noobj_loss += self.lambda_noobj * self.bce(
                            predictions[b, y, x, 0].unsqueeze(0),
                            torch.zeros(1, device=device)
                        )
        
        # Normalize by batch size
        obj_loss /= batch_size
        noobj_loss /= batch_size
        coord_loss /= batch_size
        class_loss /= batch_size
        
        # Total loss
        total_loss = obj_loss + noobj_loss + coord_loss + class_loss
        
        return total_loss, {
            'obj': obj_loss.item(),
            'noobj': noobj_loss.item(),
            'coord': coord_loss.item(),
            'class': class_loss.item()
        }


# Training function
def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_loss_components = {'obj': 0.0, 'noobj': 0.0, 'coord': 0.0, 'class': 0.0}
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, targets, _, _ in progress_bar:
        images = images.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss, loss_components = loss_fn(predictions, targets, device)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        epoch_loss += loss.item()
        for k, v in loss_components.items():
            epoch_loss_components[k] += v
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average loss
    avg_loss = epoch_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}
    
    return avg_loss, avg_components


# Validation function
def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_loss_components = {'obj': 0.0, 'noobj': 0.0, 'coord': 0.0, 'class': 0.0}
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, targets, _, _ in progress_bar:
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss, loss_components = loss_fn(predictions, targets, device)
            
            # Update statistics
            val_loss += loss.item()
            for k, v in loss_components.items():
                val_loss_components[k] += v
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average loss
    avg_loss = val_loss / len(val_loader)
    avg_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
    
    return avg_loss, avg_components


# Non-maximum suppression for inference
def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """
    Perform non-maximum suppression on YOLO predictions
    
    Args:
        predictions: Model predictions (B, H, W, num_classes+5)
        conf_threshold: Confidence threshold for objectness
        iou_threshold: IoU threshold for box suppression
        
    Returns:
        List of detections for each image
    """
    batch_size = predictions.shape[0]
    grid_size = predictions.shape[1]
    num_classes = predictions.shape[3] - 5
    
    all_detections = []
    
    for b in range(batch_size):
        image_detections = []
        
        # Process each grid cell
        for y in range(grid_size):
            for x in range(grid_size):
                pred = predictions[b, y, x]
                
                # Check objectness confidence
                objectness = torch.sigmoid(pred[0])
                if objectness < conf_threshold:
                    continue
                
                # Get bounding box coordinates
                box_x = (torch.sigmoid(pred[1]) + x) / grid_size
                box_y = (torch.sigmoid(pred[2]) + y) / grid_size
                box_w = pred[3] / grid_size
                box_h = pred[4] / grid_size
                
                # Get class with highest probability
                classes = pred[5:]
                class_scores = torch.softmax(classes, dim=0)
                class_id = torch.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                # Create detection
                detection = {
                    'confidence': objectness * class_conf,
                    'class_id': class_id.item(),
                    'class_conf': class_conf.item(),
                    'box': [box_x.item(), box_y.item(), box_w.item(), box_h.item()]
                }
                
                image_detections.append(detection)
        
        # Sort detections by confidence
        image_detections = sorted(image_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS
        nms_detections = []
        while image_detections:
            best_detection = image_detections.pop(0)
            nms_detections.append(best_detection)
            
            # Remove detections with high IoU
            image_detections = [
                d for d in image_detections
                if calculate_iou(best_detection['box'], d['box']) < iou_threshold
            ]
        
        all_detections.append(nms_detections)
    
    return all_detections


# Calculate IoU between two boxes
def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection area
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


# Visualize predictions
def visualize_predictions(image, detections, age_categories, output_path=None):
    """
    Visualize YOLO predictions on an image
    
    Args:
        image: PIL Image
        detections: List of detections from non_max_suppression
        age_categories: List of age category ranges
        output_path: Path to save the visualization
    """
    # Convert tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
        # Convert to PIL
        image = transforms.ToPILImage()(image)
    
    # Create a copy for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Original image dimensions
    width, height = image.size
    
    # Colors for different age categories
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green (dark)
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128)   # Purple
    ]
    
    # Draw each detection
    for detection in detections:
        box = detection['box']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Convert box from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x_center, y_center, box_width, box_height = box
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Draw bounding box
        color = colors[class_id % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Get age range label
        age_range = age_categories[class_id]
        label = f"Age {age_range[0]}-{age_range[1]}: {confidence:.2f}"
        
        # Draw label
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 5, y1 - 15), label, fill=(255, 255, 255))
    
    # Save or return the annotated image
    if output_path:
        draw_image.save(output_path)
    
    return draw_image


# Main training function
def train_yolo_age_detector(
    num_epochs=30,
    batch_size=16,
    learning_rate=0.001,
    img_size=416,
    save_path='models/yolo_age_detector.pth'
):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_age_prediction_dataloaders(
        batch_size=batch_size,
        img_size=img_size,
        detect_faces=True,
        num_workers=0  # Adjust based on your system
    )
    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = YOLOAgeDetector(num_classes=11)  # 11 age categories
    model = model.to(device)
    
    # Create loss function
    loss_fn = YOLOLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Track training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create directory for saving models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_components = validate(
            model, val_loader, loss_fn, device
        )
        val_losses.append(val_loss)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} - "
              f"Obj: {train_components['obj']:.4f}, "
              f"NoObj: {train_components['noobj']:.4f}, "
              f"Coord: {train_components['coord']:.4f}, "
              f"Class: {train_components['class']:.4f}")
        
        print(f"Val Loss: {val_loss:.4f} - "
              f"Obj: {val_components['obj']:.4f}, "
              f"NoObj: {val_components['noobj']:.4f}, "
              f"Coord: {val_components['coord']:.4f}, "
              f"Class: {val_components['class']:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_components': train_components,
                'val_components': val_components
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    
    return model


# Test inference
def test_inference(model_path, test_loader, device, age_categories, output_dir='predictions'):
    """
    Test the model on some test images and visualize predictions
    """
    # Load model
    model = YOLOAgeDetector(num_classes=11)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on test images
    with torch.no_grad():
        for batch_idx, (images, targets, original_info, paths) in enumerate(test_loader):
            if batch_idx >= 10:  # Process only 10 batches
                break
                
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Apply NMS to get detections
            detections = non_max_suppression(predictions)
            
            # Process each image in the batch
            for i, (image, image_detections, path) in enumerate(zip(images, detections, paths)):
                # Move image tensor to CPU and convert to PIL
                image_cpu = image.cpu()
                
                # Visualize predictions
                vis_image = visualize_predictions(
                    image_cpu, 
                    image_detections,
                    age_categories,
                    output_path=os.path.join(output_dir, f"pred_{batch_idx}_{i}.jpg")
                )
    
    print(f"Saved predictions to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Age categories
    age_categories = [
        (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
        (27, 32), (38, 42), (38, 43), (38, 48),
        (48, 53), (60, 100)
    ]
    
    # Train model
    model = train_yolo_age_detector(
        num_epochs=30,
        batch_size=16,
        learning_rate=0.001,
        save_path='models/yolo_age_detector.pth'
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test dataloader (using validation set for demonstration)
    _, test_loader = get_age_prediction_dataloaders(
        batch_size=4,
        img_size=416,
        detect_faces=True
    )
    
    # Test inference
    test_inference(
        model_path='models/yolo_age_detector.pth',
        test_loader=test_loader,
        device=device,
        age_categories=age_categories,
        output_dir='predictions'
    )