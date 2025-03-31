import os
import shutil
import torch
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import tarfile
import pandas as pd
import torchvision.transforms as transforms
import re
import cv2

# Parameters with default values matching your structure
BASE_DIR = "data"
FACES_ARCHIVE = os.path.join(BASE_DIR, "faces.tar.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "age_dataset")
FOLD_FILES = [
    os.path.join(BASE_DIR, "fold_0_data.txt"),
    os.path.join(BASE_DIR, "fold_1_data.txt"),
    os.path.join(BASE_DIR, "fold_2_data.txt"),
    os.path.join(BASE_DIR, "fold_3_data.txt"),
    os.path.join(BASE_DIR, "fold_4_data.txt")
]

# Age categories as defined in your original code
AGE_CATEGORIES = [
    (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), 
    (27, 32), (38, 42), (38, 43), (38, 48), 
    (48, 53), (60, 100)
]

def extract_faces_archive(archive_path, extract_to):
    """Extract the faces archive if it hasn't been already"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        print(f"Extracting {archive_path} to {extract_to}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Extract with progress tracking
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting faces"):
                tar.extract(member, extract_to)
        print("Extraction complete.")
    else:
        print(f"Directory {extract_to} already exists. Skipping extraction.")

def get_age_class(age_info):
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
                for i, category in enumerate(AGE_CATEGORIES):
                    if age_range == category:
                        return i
        
        # Try to parse as a specific age
        else:
            age = int(age_info)
            
            # Find appropriate category based on the specific age
            for i, (lower, upper) in enumerate(AGE_CATEGORIES):
                if lower <= age <= upper:
                    return i
        
        # If no match found
        return -1
        
    except:
        return -1

def load_fold_data(fold_files):
    """Load metadata from fold files"""
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
            df['age_class'] = df['age'].apply(get_age_class)
            
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

def get_image_path(row, faces_dir):
    """Get full path to the face image"""
    user_id = row['user_id']
    face_id = row['face_id']
    original_image = row['original_image']
    
    # Format: coarse_tilt_aligned_face.{face_id}.{original_image_name}
    filename = f"coarse_tilt_aligned_face.{face_id}.{original_image}"
    
    # Fix path separators for the system
    path = os.path.join(faces_dir, str(user_id), filename)
    return os.path.normpath(path)

def create_yolo_dataset(
    faces_dir,
    fold_files,
    output_dir,
    train_folds=[0, 1, 2, 3],
    val_fold=4,
    img_size=416
):
    """
    Create a YOLO format dataset from the original face dataset
    
    Args:
        faces_dir: Directory containing extracted face images
        fold_files: List of fold data files
        output_dir: Directory to save the YOLO format dataset
        train_folds: List of fold numbers to use for training
        val_fold: Fold number to use for validation
        img_size: Size to resize images to
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train and val directories
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')
    
    for directory in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load all fold data
    all_data = load_fold_data(fold_files)
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Create classes.txt file with age categories
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for i, (lower, upper) in enumerate(AGE_CATEGORIES):
            f.write(f"age_{lower}_{upper}\n")
    
    # Split data into train and validation sets
    train_indices = [i for i, file in enumerate(fold_files) if i in train_folds]
    val_indices = [i for i, file in enumerate(fold_files) if i == val_fold]
    
    train_files = [fold_files[i] for i in train_indices]
    val_files = [fold_files[i] for i in val_indices]
    
    train_data = load_fold_data(train_files)
    val_data = load_fold_data(val_files)
    
    print(f"Processing {len(train_data)} training images...")
    process_dataset(train_data, faces_dir, train_img_dir, train_label_dir, img_size, transform)
    
    print(f"Processing {len(val_data)} validation images...")
    process_dataset(val_data, faces_dir, val_img_dir, val_label_dir, img_size, transform)
    
    # Create data.yaml file
    create_data_yaml(output_dir, len(AGE_CATEGORIES))
    
    print(f"YOLO dataset created successfully at {output_dir}")

def detect_face(image_np):
    """
    Detect faces in an image using OpenCV's Haar cascade classifier
    
    Args:
        image_np: Image as NumPy array (BGR format)
        
    Returns:
        Tuple (x, y, w, h) of face or None if no face detected
    """
    # Try to find the Haar cascade file
    cascade_paths = [
        # Common paths for haarcascade file
        'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    ]

    cascade_file = None
    for path in cascade_paths:
        if os.path.exists(path):
            cascade_file = path
            break
    
    if cascade_file is None:
        print("Warning: Could not find face cascade file. Using fallback bounding box.")
        return None
    
    # Initialize classifier
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    # Convert to grayscale for face detection
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If faces found, return the largest one
    if len(faces) > 0:
        # Get largest face by area
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return largest_face
    
    return None

def process_dataset(data, faces_dir, img_dir, label_dir, img_size, transform):
    """Process a dataset and save images and labels in YOLO format"""
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            # Get image path
            img_path = get_image_path(row, faces_dir)
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Get class and create a unique filename
            class_id = int(row['age_class'])  # Ensure it's an integer
            
            # Create a unique filename based on the original path
            filename = os.path.basename(img_path).replace('coarse_tilt_aligned_face.', '')
            base_filename = f"{idx}_{filename.split('.')[0]}"  # Add index for uniqueness
            
            # Load image as PIL Image
            with Image.open(img_path).convert('RGB') as img:
                # Save original dimensions
                orig_width, orig_height = img.size
                
                # Convert PIL image to OpenCV format for face detection
                img_np = np.array(img)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Detect face using OpenCV
                face_coords = detect_face(img_cv)
                
                # Apply transformations for saving
                img_tensor = transform(img)
                
                # Convert tensor back to PIL for saving
                img_save = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_save)
                
                # Save image
                save_path = os.path.join(img_dir, f"{base_filename}.jpg")
                img_pil.save(save_path)
                
                # Get bounding box coordinates (either from face detection or fallback)
                if face_coords is not None:
                    x, y, w, h = face_coords
                    
                    # Convert to YOLO format (normalized center coordinates + width/height)
                    x_center = (x + w/2) / orig_width
                    y_center = (y + h/2) / orig_height
                    width_norm = w / orig_width
                    height_norm = h / orig_height
                else:
                    # Fallback: use a bounding box covering 80% of the image
                    x_center, y_center = 0.5, 0.5
                    width_norm, height_norm = 0.8, 0.8
                
                # Clamp values to ensure they're within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0.05, min(1, width_norm))
                height_norm = max(0.05, min(1, height_norm))
                
                # Save label in YOLO format
                label_path = os.path.join(label_dir, f"{base_filename}.txt")
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        except Exception as e:
            print(f"Error processing image for index {idx}: {str(e)}")

def create_data_yaml(output_dir, num_classes):
    """Create a data.yaml file for YOLO training"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join('images', 'train')}\n")
        f.write(f"val: {os.path.join('images', 'val')}\n")
        f.write(f"nc: {num_classes}\n")
        f.write("names:\n")
        
        # Get class names from classes.txt
        classes_path = os.path.join(output_dir, 'classes.txt')
        with open(classes_path, 'r') as cf:
            for i, line in enumerate(cf):
                class_name = line.strip()
                f.write(f"  {i}: '{class_name}'\n")

def main():
    parser = argparse.ArgumentParser(description='Create YOLO format dataset from facial age dataset')
    parser.add_argument('--extract', action='store_true',
                        help='Extract faces archive (if needed)')
    parser.add_argument('--faces-dir', type=str, default=os.path.join(BASE_DIR, "faces"),
                        help='Directory containing face images (default: data/faces)')
    parser.add_argument('--faces-archive', type=str, default=FACES_ARCHIVE,
                        help='Path to faces archive file (default: data/faces.tar.gz)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Directory to save the YOLO format dataset (default: data/age_dataset)')
    parser.add_argument('--fold-files', type=str, nargs='+', default=FOLD_FILES,
                        help='List of fold data files')
    parser.add_argument('--train-folds', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Fold numbers to use for training (default: [0, 1, 2, 3])')
    parser.add_argument('--val-fold', type=int, default=4,
                        help='Fold number to use for validation (default: 4)')
    parser.add_argument('--img-size', type=int, default=416,
                        help='Size to resize images to (default: 416)')
    parser.add_argument('--cascade-file', type=str, default=None,
                        help='Path to haarcascade_frontalface_default.xml file (optional)')
    
    args = parser.parse_args()
    
    # If cascade file is provided, check if it exists
    if args.cascade_file:
        if not os.path.exists(args.cascade_file):
            print(f"Warning: Provided cascade file {args.cascade_file} does not exist.")
        else:
            # Add to the beginning of cascade_paths in detect_face
            global cascade_paths
            cascade_paths = [args.cascade_file]
    
    # Extract faces archive if requested
    if args.extract:
        extract_faces_archive(args.faces_archive, args.faces_dir)
    
    # Check if OpenCV is available
    try:
        import cv2
        print("OpenCV is available. Face detection will be used.")
    except ImportError:
        print("OpenCV is not available. Face detection will be disabled.")
        print("Please install OpenCV: pip install opencv-python")
        return
    
    create_yolo_dataset(
        faces_dir=args.faces_dir,
        fold_files=args.fold_files,
        output_dir=args.output,
        train_folds=args.train_folds,
        val_fold=args.val_fold,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()