import os
import cv2
import numpy as np
import joblib
from PIL import Image
from tqdm import tqdm
import csv

# --- 1. CONFIGURATION ---

# The folder of NEW images you want to process
INPUT_FOLDER = "processed_images" 

# Folder to save the final highlighted images and CSV
FINAL_OUTPUT_DIR = "PROJECT_FINAL_OUTPUT"
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# Path to the model and features
# <-- MODIFIED: Pointing to your best tuned model
MODEL_FILE = os.path.join("final_model", "wildlife_classifier_xgb.joblib")
# <-- MODIFIED: Pointing to your feature script
FEATURE_SCRIPT_FILE = "featuredetect.py" 

# --- 2. LOAD FEATURE FUNCTIONS FROM STEP 2 SCRIPT ---
print("Importing feature functions from featuredetect.py...")
import sys
sys.path.append(os.getcwd())

try:
    # <-- MODIFIED: Importing from your file 'featuredetect.py'
    from feature_extraction import (
        extract_all_features,
        get_color_features,
        get_lbp_features,
        get_hog_features
    )
    # We also need the LBP parameters from that file
    from feature_extraction import LBP_POINTS, LBP_RADIUS
except ImportError as e:
    # <-- MODIFIED: Corrected error message
    print(f"Error: Could not import from 'featuredetect.py'.")
    print(f"Make sure that file is in the same directory.")
    print(f"Details: {e}")
    exit()

# We also need the preprocessing function from Step 1
try:
    # <-- MODIFIED: Importing from your file 'scaling.py'
    from image_processing import center_crop_to_4by3
except ImportError as e:
    # <-- MODIFIED: Corrected error message
    print(f"Error: Could not import from 'scaling.py'.")
    exit()

print("Feature functions imported successfully.")

# --- 3. CONSTANTS & HELPER FUNCTIONS ---
CELL_W = 100
CELL_H = 75
GRID_COLS = 8
GRID_ROWS = 8
TARGET_SIZE = (800, 600)

# --- This function handles Rule #1 (Preprocessing) ---
def preprocess_image_for_pipeline(image_path):
    """
    Loads, crops, and resizes a single image based on all rules.
    Returns an 800x600 OpenCV (BGR) image, or None if invalid.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Rule 1a: CROP
            img_4by3 = center_crop_to_4by3(img)
            width, height = img_4by3.size

            # Rule 1b/1c: Resize or Discard
            if width == 800 and height == 600:
                final_pil_img = img_4by3
            elif width > 800:
                final_pil_img = img_4by3.resize(TARGET_SIZE, Image.LANCZOS)
            else:
                return None # Discard if too small

            # Convert from PIL (RGB) to OpenCV (BGR)
            open_cv_image = np.array(final_pil_img) 
            open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB to BGR
            return open_cv_image

    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

# --- This function adds the green highlight ---
def draw_highlight(image, r, c):
    """Draws a semi-transparent green box on the image."""
    y1, y2 = r * CELL_H, (r + 1) * CELL_H
    x1, x2 = c * CELL_W, (c + 1) * CELL_W
    
    # Create a green overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1) # Green
    
    # Blend the overlay with the original image
    alpha = 0.3 # Transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# --- 4. MAIN PIPELINE ---
if __name__ == "__main__":
    
    print(f"\nLoading trained model: {MODEL_FILE}")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        # <-- MODIFIED: Corrected error message
        print(f"Error: Model file '{MODEL_FILE}' not found. Did you run the training script?")
        exit()
    
    # --- Setup for the CSV file (Rule 4b) ---
    csv_path = os.path.join(FINAL_OUTPUT_DIR, "output.csv")
    # Create the header row (ImageFileName, c01, c02, ..., c64)
    header = ["ImageFileName"]
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            header.append(f"c{r*GRID_COLS + c + 1:02d}") # Formats as c01, c02...

    # --- Start processing images ---
    print(f"Opening CSV file for writing: {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) # Write the header
        
        print(f"Processing all images in: {INPUT_FOLDER}...")
        for filename in tqdm(os.listdir(INPUT_FOLDER)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(INPUT_FOLDER, filename)
            
            # --- 1. Preprocess the image (Rules 1a, 1b, 1c) ---
            image_800x600 = preprocess_image_for_pipeline(image_path)
            
            if image_800x600 is None:
                continue # Image was discarded (too small)

            # This will be the list for our CSV row
            csv_row = [filename]
            
            # --- 2. Loop through 8x8 Grid ---
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    y1, y2 = r * CELL_H, (r + 1) * CELL_H
                    x1, x2 = c * CELL_W, (c + 1) * CELL_W
                    
                    # --- 3. Extract Features ---
                    cell = image_800x600[y1:y2, x1:x2]
                    try:
                        # Note: This function no longer needs a prefix
                        features = extract_all_features(cell) 
                    except Exception as e:
                        print(f"Warning: Feature extraction failed for {filename} cell ({r},{c}). Skipping cell.")
                        csv_row.append(0) # Default to 0
                        continue
                    
                    # Reshape for a single prediction
                    features = features.reshape(1, -1)
                    
                    # --- 4. Get Model Prediction ---
                    prediction = model.predict(features)[0] # Get the 0 or 1
                    csv_row.append(prediction)
                    
                    # --- 5. Highlight Image (Rule 4a) ---
                    if prediction == 1:
                        draw_highlight(image_800x600, r, c)
            
            # --- 6. Save Outputs for this Image ---
            # Save the CSV row
            writer.writerow(csv_row)
            
            # Save the (possibly highlighted) image
            output_image_path = os.path.join(FINAL_OUTPUT_DIR, f"HIGHLIGHTED_{filename}")
            cv2.imwrite(output_image_path, image_800x600)

    print("\n--- FINAL PIPELINE COMPLETE! ---")
    print(f"All outputs saved to: {FINAL_OUTPUT_DIR}")