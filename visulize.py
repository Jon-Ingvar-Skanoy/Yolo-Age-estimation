import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Import your updated dataloader
from dataset import FacialAgeDataset, get_age_prediction_dataloaders

def visualize_detected_faces(root_dir='data/faces', output_dir='face_detection_results'):
    """
    Visualize the detected face bounding boxes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Age category mapping
    age_categories = [
        (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), 
        (27, 32), (38, 42), (38, 43), (38, 48), 
        (48, 53), (60, 100)
    ]
    
    # Create dataset without transforms to visualize original images
    fold_files_base = os.path.dirname(root_dir)
    fold_file = os.path.join(fold_files_base, 'fold_0_data.txt')
    
    # Create dataset with face detection enabled and no workers
    # Initialize with dataset - no DataLoader to avoid any pickling issues
    dataset = FacialAgeDataset(
        root_dir=root_dir,
        fold_files=[fold_file],
        transform=None,  # No transforms to keep original
        detect_faces=True
    )
    
    # Get a subset of samples to visualize
    num_samples = min(20, len(dataset))
    indices = list(range(min(num_samples, len(dataset))))  # Use first N samples for consistency
    
    # Process each sample
    for i, idx in enumerate(indices):
        try:
            # Get the sample
            sample = dataset[idx]
            img = sample['image']
            target = sample['targets']
            info = sample['original_info']
            path = sample['path']
            
            # Check if image is valid
            if not isinstance(img, Image.Image):
                print(f"Invalid image at index {idx}")
                continue
            
            # Create a copy of the image for drawing
            img_with_box = img.copy()
            draw = ImageDraw.Draw(img_with_box)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Get the class ID and label
            class_id = int(target[0, 0])
            if 0 <= class_id < len(age_categories):
                label = f"Age: {age_categories[class_id][0]}-{age_categories[class_id][1]}"
            else:
                label = f"Class {class_id}"
            
            # Draw detected face if available
            if info.get('face_coords'):
                x, y, w, h = info['face_coords']
                draw.rectangle([x, y, x+w, y+h], outline="lime", width=3)
                
                # Add label
                draw.rectangle([x, y-20, x+150, y], fill="lime")
                draw.text((x+5, y-15), f"Detected: {label}", fill="black")
                
                print(f"Image {i} - Found face: x={x}, y={y}, w={w}, h={h}")
            else:
                # Draw fallback box (80% of image)
                margin = int(min(img_width, img_height) * 0.1)
                draw.rectangle([margin, margin, img_width-margin, img_height-margin], 
                               outline="yellow", width=3)
                
                # Add label
                draw.rectangle([margin, margin-20, margin+180, margin], fill="yellow")
                draw.text((margin+5, margin-15), f"Default: {label}", fill="black")
                
                print(f"Image {i} - No face detected, using default box")
            
            # Also draw normalized target box
            x_center, y_center, width, height = target[0, 1:].tolist()
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            # Draw YOLO target box in red
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Add YOLO target label
            draw.rectangle([x1, y2, x1+150, y2+20], fill="red")
            draw.text((x1+5, y2+5), f"YOLO Target", fill="white")
            
            # Save the image
            output_file = os.path.join(output_dir, f"face_detection_{i}.png")
            img_with_box.save(output_file)
            print(f"Saved {output_file}")
            
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
    
    print("Face detection visualization complete")

def visualize_transformed_batch(root_dir='data/faces', output_dir='transformed_faces'):
    """
    Visualize a batch of transformed images with detected face boxes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Age category mapping
    age_categories = [
        (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), 
        (27, 32), (38, 42), (38, 43), (38, 48), 
        (48, 53), (60, 100)
    ]
    
    # Get dataloader with small batch size and no workers
    train_loader, _ = get_age_prediction_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        detect_faces=True,
        num_workers=0  # Use 0 workers to avoid multiprocessing issues
    )
    
    # Get a batch
    try:
        data_iter = iter(train_loader)
        batch = next(data_iter)
        
        # Check if we have a valid batch
        if len(batch) != 4:
            print(f"Unexpected batch structure: {len(batch)} elements")
            return
            
        images, targets, original_info, paths = batch
        
        # Process each image
        for idx, (img_tensor, target, info) in enumerate(zip(images, targets, original_info)):
            # Denormalize the image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = img_denorm.clamp(0, 1)
            
            # Convert to PIL
            img_pil = transforms.ToPILImage()(img_denorm)
            draw = ImageDraw.Draw(img_pil)
            
            # Get image dimensions
            img_width, img_height = img_pil.size
            
            # Draw YOLO format box
            if len(target) > 0:
                class_id, x_center, y_center, width, height = target[0]
                
                # Convert normalized coordinates to pixel values
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Add label
                class_id_int = int(class_id)
                if 0 <= class_id_int < len(age_categories):
                    label = f"Age: {age_categories[class_id_int][0]}-{age_categories[class_id_int][1]}"
                else:
                    label = f"Class {class_id_int}"
                
                draw.rectangle([x1, y1-20, x1+150, y1], fill="red")
                draw.text((x1+5, y1-15), label, fill="white")
                
                # Add detection method
                if info.get('face_coords'):
                    method = "Face detected"
                else:
                    method = "Default box"
                
                draw.text((10, 10), method, fill="white", stroke_width=1, stroke_fill="black")
            
            # Save the image
            output_file = os.path.join(output_dir, f"transformed_{idx}.png")
            img_pil.save(output_file)
            print(f"Saved {output_file}")
    
    except Exception as e:
        print(f"Error in batch visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Transformed batch visualization complete")

def visualize_yolo_training_format(root_dir='data/faces', output_dir='yolo_format'):
    """
    Visualize samples in the YOLO training format with class and bounding box overlays
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Age category mapping
    age_categories = [
        (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), 
        (27, 32), (38, 42), (38, 43), (38, 48), 
        (48, 53), (60, 100)
    ]
    
    # Create a transform pipeline similar to what would be used in training
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Standard YOLO input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with transforms
    fold_files_base = os.path.dirname(root_dir)
    fold_file = os.path.join(fold_files_base, 'fold_0_data.txt')
    
    dataset = FacialAgeDataset(
        root_dir=root_dir,
        fold_files=[fold_file],
        transform=transform,
        detect_faces=True
    )
    
    # Create dataloader with no workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: ({
            'image': torch.stack([s['image'] for s in x]),
            'targets': [s['targets'] for s in x],
            'original_info': [s.get('original_info', {}) for s in x],
            'path': [s.get('path', '') for s in x]
        })
    )
    
    # Get a batch
    try:
        batch = next(iter(dataloader))
        images = batch['image']
        targets = batch['targets']
        original_info = batch['original_info']
        paths = batch['path']
        
        # Create a grid of images (4x4)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        # Process each image
        for i, (img_tensor, target, info) in enumerate(zip(images, targets, original_info)):
            if i >= 16:  # Limit to 16 images for the grid
                break
                
            # Denormalize the image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = img_denorm.clamp(0, 1)
            
            # Convert to numpy for plotting
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            # Plot the image
            axes[i].imshow(img_np)
            
            # Draw bounding box
            if len(target) > 0:
                class_id, x_center, y_center, width, height = target[0]
                
                # Convert normalized coordinates to pixel values
                img_height, img_width = img_np.shape[:2]
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
                
                # Add label
                class_id_int = int(class_id)
                if 0 <= class_id_int < len(age_categories):
                    label = f"Age: {age_categories[class_id_int][0]}-{age_categories[class_id_int][1]}"
                else:
                    label = f"Class {class_id_int}"
                
                axes[i].text(x1, y1-5, label, color='white', 
                           bbox=dict(facecolor='red', alpha=0.7))
            
            # Remove axis ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Hide any unused subplots
        for j in range(i+1, 16):
            axes[j].axis('off')
        
        # Save the grid
        plt.tight_layout()
        grid_file = os.path.join(output_dir, "yolo_grid.png")
        plt.savefig(grid_file)
        plt.close(fig)
        print(f"Saved grid to {grid_file}")
        
        # Also save individual images
        for i, (img_tensor, target) in enumerate(zip(images[:8], targets[:8])):
            # Denormalize and convert to PIL
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_denorm = img_tensor * std + mean
            img_denorm = img_denorm.clamp(0, 1)
            img_pil = transforms.ToPILImage()(img_denorm)
            
            # Draw on the image
            draw = ImageDraw.Draw(img_pil)
            img_width, img_height = img_pil.size
            
            # Draw bounding box
            if len(target) > 0:
                class_id, x_center, y_center, width, height = target[0]
                
                # Convert normalized coordinates to pixel values
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Add label
                class_id_int = int(class_id)
                if 0 <= class_id_int < len(age_categories):
                    label = f"Age: {age_categories[class_id_int][0]}-{age_categories[class_id_int][1]}"
                else:
                    label = f"Class {class_id_int}"
                
                draw.rectangle([x1, y1-20, x1+150, y1], fill="red")
                draw.text((x1+5, y1-15), label, fill="white")
            
            # Save the image
            indiv_file = os.path.join(output_dir, f"yolo_sample_{i}.png")
            img_pil.save(indiv_file)
            print(f"Saved individual sample to {indiv_file}")
            
    except Exception as e:
        print(f"Error in YOLO visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("YOLO format visualization complete")

if __name__ == "__main__":
    # Run all visualization methods
    print("Starting face detection visualization...")
    visualize_detected_faces()
    
    print("\nStarting transformed batch visualization...")
    visualize_transformed_batch()
    
    print("\nStarting YOLO format visualization...")
    visualize_yolo_training_format()