import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import re
import time
class FacialAgeDataset(Dataset):
    def __init__(self, root_dir, fold_files, transform=None, detect_faces=True):
        """
        Args:
            root_dir (string): Root directory with all the face images
            fold_files (list): List of fold data files to include
            transform (callable, optional): Optional transform to be applied on a sample
            detect_faces (bool): Whether to use OpenCV's face detector to create bounding boxes
        """
        self.root_dir = root_dir
        self.transform = transform
        self.detect_faces = detect_faces
        
        # Age categories to class indices (0-based)
        self.age_categories = [
            (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), 
            (27, 32), (38, 42), (38, 43), (38, 48), 
            (48, 53), (60, 100)
        ]
        
        # Load metadata from fold files
        self.metadata = self._load_metadata(fold_files)
        
        # DO NOT initialize the face detector here - will be done in __getitem__
        # This makes the dataset picklable for use with multiple workers
        self.face_detector = None
        
    def _load_metadata(self, fold_files):
        """Load and parse metadata from fold files"""
        all_data = []
        
        column_names = [
            'user_id', 'original_image', 'face_id', 'age', 'gender', 
            'x', 'y', 'dx', 'dy', 'tilt_ang', 'fiducial_yaw_angle', 'fiducial_score'
        ]
        
        for fold_file in fold_files:
            try:
                # Load the fold file
                df = pd.read_csv(fold_file, sep='\t', header=None, names=column_names)
                
                # Process age information
                df['age_class'] = df['age'].apply(self._get_age_class)
                
                # Filter out entries with invalid age classes
                df = df[df['age_class'] != -1]
                
                # Add fold data to our list
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading fold file {fold_file}: {str(e)}")
        
        # Combine all data
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame(columns=column_names + ['age_class'])
    
    def _get_age_class(self, age_info):
        """
        Convert age information to appropriate class index
        Age info can be either a specific age or a range like "(25, 32)"
        
        Returns:
            int: The class index (0-based) or -1 if cannot be classified
        """
        try:
            # Try to parse as a range
            if isinstance(age_info, str) and '(' in age_info:
                # Extract the numbers from the string using regex
                match = re.findall(r'\d+', age_info)
                if len(match) >= 2:
                    lower, upper = int(match[0]), int(match[1])
                    age_range = (lower, upper)
                    
                    # Find matching category
                    for i, category in enumerate(self.age_categories):
                        if age_range == category:
                            return i
            
            # Try to parse as a specific age
            else:
                age = int(age_info)
                
                # Find appropriate category based on the specific age
                for i, (lower, upper) in enumerate(self.age_categories):
                    if lower <= age <= upper:
                        return i
            
            # If no match found
            return -1
            
        except:
            return -1
    
    def _get_image_path(self, row):
        """Get full path to the face image"""
        user_id = row['user_id']
        face_id = row['face_id']
        original_image = row['original_image']
        
        # Format: coarse_tilt_aligned_face.{face_id}.{original_image_name}
        filename = f"coarse_tilt_aligned_face.{face_id}.{original_image}"
        
        # Fix path separators for the system
        path = os.path.join(self.root_dir, str(user_id), filename)
        return os.path.normpath(path)
    
    def _init_face_detector(self):
        """
        Initialize the face detector on demand.
        This avoids pickling issues with multiprocessing.
        """
        if self.face_detector is None and self.detect_faces:
            try:
                import cv2
                
                # Try to find the Haar cascade file in standard locations
                cascade_file = None
                possible_paths = [
                    # Standard OpenCV installation paths
                    'haarcascade_frontalface_default.xml',
                    'data/haarcascades/haarcascade_frontalface_default.xml',
                    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'),
                    # Add more potential paths if needed
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        cascade_file = path
                        break
                
                if cascade_file:
                    self.face_detector = cv2.CascadeClassifier(cascade_file)
                    print(f"Face detector initialized with {cascade_file}")
                else:
                    print("Warning: Could not find face cascade file. Using default bounding boxes.")
                    self.detect_faces = False
            except ImportError:
                print("Warning: OpenCV (cv2) not available. Face detection disabled.")
                self.detect_faces = False
    
    def _detect_face(self, image):
        """
        Detect face in the image using OpenCV's Haar cascade
        
        Args:
            image: PIL Image
            
        Returns:
            x, y, w, h: Coordinates of detected face, or None if no face detected
        """
        if not self.detect_faces:
            return None
        
        # Initialize detector if needed
        self._init_face_detector()
        
        if self.face_detector is None:
            return None

        try:
            import cv2
            
            # Convert PIL Image to OpenCV format (numpy array)
            img_cv = np.array(image)
            
            # Convert to grayscale for face detection
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_cv
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # If faces found, return the largest one
            if len(faces) > 0:
                # Get largest face by area
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                return x, y, w, h
            
            return None
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict: A dictionary containing:
                'image': The image tensor
                'targets': YOLO format targets [class_id, x_center, y_center, width, height]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.metadata.iloc[idx]
        
        # Get image path and load image
        img_path = self._get_image_path(row)
        
        try:
            # Load the image
            image = Image.open(img_path).convert('RGB')
            orig_width, orig_height = image.size
            
            # Get the age class
            age_class = row['age_class']
            
            # Try to detect face
            face_coords = self._detect_face(image)
            
            # If face detection is enabled and a face was found, use those coordinates
            if face_coords:
                x, y, w, h = face_coords
                
                # Create YOLO format target (before transform)
                # Convert to center coordinates and normalize
                x_center = (x + w/2) / orig_width
                y_center = (y + h/2) / orig_height
                width_norm = w / orig_width
                height_norm = h / orig_height
            else:
                # Fallback: use a bounding box covering 80% of the image
                # This is more reasonable than full image for aligned face images
                margin = 0.1  # 10% margin on each side
                x_center = 0.5
                y_center = 0.5
                width_norm = 0.8
                height_norm = 0.8
            
            # Store original info
            original_info = {
                'width': orig_width,
                'height': orig_height,
                'face_coords': face_coords
            }
            
            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)
            
            # Create YOLO format target
            yolo_target = torch.zeros((1, 5))
            yolo_target[0, 0] = age_class  # class index
            yolo_target[0, 1] = x_center
            yolo_target[0, 2] = y_center
            yolo_target[0, 3] = width_norm
            yolo_target[0, 4] = height_norm
            
            return {
                'image': image,
                'targets': yolo_target,
                'original_info': original_info,
                'path': img_path
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default sample in case of error
            if self.transform:
                default_image = self.transform(Image.new('RGB', (224, 224), color='gray'))
            else:
                default_image = torch.zeros((3, 224, 224))
            
            return {
                'image': default_image,
                'targets': torch.zeros((1, 5)),
                'original_info': {'width': 224, 'height': 224, 'face_coords': None},
                'path': img_path
            }


def get_age_prediction_dataloaders(
    root_dir='data/faces',
    train_folds=[0, 1, 2, 3],
    val_fold=4,
    batch_size=16,
    num_workers=0,  # Default to 0 workers due to face detection
    img_size=416,  # YOLO typically uses 416x416
    detect_faces=True
):
    """
    Create train and validation dataloaders for the age prediction task
    
    Args:
        root_dir: Root directory containing the dataset
        train_folds: List of fold numbers to use for training
        val_fold: Fold number to use for validation
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders (default reduced for face detection)
        img_size: Size to resize images to (YOLO typically uses 416x416)
        detect_faces: Whether to use face detection for bounding boxes
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create file paths for fold data
    fold_files_base = os.path.dirname(root_dir)  # Ensure we get parent directory correctly
    
    # Training dataset
    train_fold_files = [os.path.join(fold_files_base, f'fold_{fold}_data.txt') for fold in train_folds]
    train_dataset = FacialAgeDataset(
        root_dir=root_dir,
        fold_files=train_fold_files,
        transform=transform,
        detect_faces=detect_faces
    )
    
    # Validation dataset
    val_fold_file = [os.path.join(fold_files_base, f'fold_{val_fold}_data.txt')]
    val_dataset = FacialAgeDataset(
        root_dir=root_dir,
        fold_files=val_fold_file,
        transform=transform,
        detect_faces=detect_faces
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=yolo_collate_fn
    )
    
    return train_loader, val_loader

def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO-formatted data
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        images: Tensor of images [batch_size, channels, height, width]
        targets: List of tensors containing target information
        original_info: List of dictionaries with original image information
        paths: List of image paths
    """
    images = []
    targets = []
    original_info = []
    paths = []
    
    for sample in batch:
        images.append(sample['image'])
        targets.append(sample['targets'])
        original_info.append(sample.get('original_info', {}))
        paths.append(sample.get('path', ''))
    
    # Stack images into a single tensor
    images = torch.stack(images, 0)
    
    return images, targets, original_info, paths


# Example usage
if __name__ == "__main__":
    # Get dataloaders with reduced workers
    train_loader, val_loader = get_age_prediction_dataloaders(
        detect_faces=True,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Print dataset statistics
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Check a batch
    start_time = time.time()
    for images, targets, original_info, paths in train_loader:
       
        
        # Print face detection results for first image
        if original_info[0].get('face_coords'):
            a = 2
       
    end_time = time.time()
    print("time passed", end_time - start_time)    