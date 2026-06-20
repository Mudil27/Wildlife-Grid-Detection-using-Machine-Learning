import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
INPUT_FOLDER = "processed_images" 
OUTPUT_DIR = "feature_data"
# VISUAL_OUTPUT_FOLDER has been removed

os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_W = 100
CELL_H = 75
GRID_COLS = 8
GRID_ROWS = 8
LBP_POINTS = 32 # Your change
LBP_RADIUS = 3

# --- 2. HAND-CRAFTED FEATURE FUNCTIONS ---

def get_color_features(cell):
    cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([cell_hsv], [0, 1, 2], None, [8, 8, 8], 
                          [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def get_lbp_features(cell):
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(cell_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, LBP_POINTS + 3),
                             range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # --- MODIFIED: Only return the histogram ---
    return hist

def get_hog_features(cell):
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # --- MODIFIED: Set visualize=False ---
    features = hog(cell_gray, orientations=8, 
                   pixels_per_cell=(8, 8), # Your change
                   cells_per_block=(1, 1), 
                   visualize=False, # Changed to False
                   feature_vector=True)
    return features

# --- REMOVED save_color_histograms function ---

def extract_all_features(cell):
    # --- MODIFIED: Removed all visualization logic ---
    
    # Color
    color_feats = get_color_features(cell)
    # LBP
    lbp_feats = get_lbp_features(cell)
    # HOG
    hog_feats = get_hog_features(cell)
    
    # --- REMOVED: Save visualization examples block ---

    combined_features = np.hstack([color_feats, lbp_feats, hog_feats])
    return combined_features

# --- 3. MAIN PROCESSING FUNCTION ---
def create_feature_matrix(image_filenames, set_name):
    all_features_list = []
    metadata_list = []
    print(f"\nProcessing {set_name} set: {len(image_filenames)} images...")
    
    for img_index, filename in enumerate(tqdm(image_filenames)):
        img_path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(img_path)
        if image is None: continue
            
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y1, y2 = r * CELL_H, (r + 1) * CELL_H
                x1, x2 = c * CELL_W, (c + 1) * CELL_W
                cell = image[y1:y2, x1:x2]
                
                try:
                    # --- MODIFIED: Removed all visualization logic ---
                    features = extract_all_features(cell)
                    all_features_list.append(features)
                    metadata_list.append((filename, r, c))
                except Exception as e:
                    print(f"\nError processing cell ({r},{c}) in {filename}: {e}")

    feature_matrix = np.array(all_features_list)
    return feature_matrix, metadata_list

# --- 4. SCRIPT EXECUTION ---
if __name__ == "__main__":
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_files:
        print(f"Error: No images found in {INPUT_FOLDER}.")
    else:
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        
        X_train, meta_train = create_feature_matrix(train_files, "Training")
        print(f"\nTraining Matrix Shape: {X_train.shape}")
        joblib.dump(X_train, os.path.join(OUTPUT_DIR, "X_train_unlabeled.joblib"))
        joblib.dump(meta_train, os.path.join(OUTPUT_DIR, "meta_train.joblib"))
        
        X_test, meta_test = create_feature_matrix(test_files, "Testing")
        print(f"Testing Matrix Shape: {X_test.shape}")
        joblib.dump(X_test, os.path.join(OUTPUT_DIR, "X_test.joblib"))
        joblib.dump(meta_test, os.path.join(OUTPUT_DIR, "meta_test.joblib"))
        
        print(f"\n--- Step 2 Complete! ---")
        print(f"Feature data saved to {OUTPUT_DIR}")
        # --- MODIFIED: Removed print statement for visuals ---