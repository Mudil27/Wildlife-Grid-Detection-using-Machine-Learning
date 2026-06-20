import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION ---

# -----------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!! TEAM DECISION GOES HERE !!!!!!!!!!!!!!!!!!!!!
#
# Review the 'clustering_output/cluster_visuals' folders and
# list all cluster IDs that contain ANY wildlife (fur, feathers,
# stripes, insects, etc.).
#
# Example: WILDLIFE_CLUSTERS = [5, 18, 32]
# -----------------------------------------------------------------
WILDLIFE_CLUSTERS = [1,2,4,9,13, 0] # <-- PUT YOUR LIST OF NUMBERS HERE


# --- Input files (from previous steps) ---
FEATURE_DATA_DIR = "feature_data"
META_TRAIN_FILE = os.path.join(FEATURE_DATA_DIR, "meta_train.joblib")
META_TEST_FILE = os.path.join(FEATURE_DATA_DIR, "meta_test.joblib")

CLUSTERING_DIR = "clustering_output"
KMEANS_MODEL_FILE = os.path.join(CLUSTERING_DIR, "kmeans_model.joblib")
TRAIN_ASSIGNMENTS_FILE = os.path.join(CLUSTERING_DIR, "cluster_assignments.joblib")

# --- Output files (The final labels!) ---
OUTPUT_DIR = "final_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, "y_train.joblib")
Y_TEST_FILE = os.path.join(OUTPUT_DIR, "y_test.joblib")

# --- 2. DEFINE THE MAPPING FUNCTION ---
def create_label_from_cluster(cluster_id, wildlife_map):
    """Checks if a cluster_id is in the wildlife map."""
    if cluster_id in wildlife_map:
        return 1
    else:
        return 0

# --- 3. SCRIPT EXECUTION ---
if __name__ == "__main__":
    
    if not WILDLIFE_CLUSTERS:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: You must edit this script first.")
        print("Please open 'run_step_4_create_labels.py' and fill in the")
        print("'WILDLIFE_CLUSTERS' list with your team's decisions.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

    # Use a 'set' for much faster lookups
    wildlife_map = set(WILDLIFE_CLUSTERS)
    print(f"Labeling complete. The following clusters are '1' (Wildlife): {WILDLIFE_CLUSTERS}")

    # --- 4. PROCESS TRAINING LABELS ---
    print(f"Loading training cluster assignments...")
    # These are the cluster IDs for every cell in X_train
    train_cluster_assignments = joblib.load(TRAIN_ASSIGNMENTS_FILE)
    
    # Loop through all assignments and convert to 0s and 1s
    y_train = [create_label_from_cluster(cid, wildlife_map) for cid in train_cluster_assignments]
    y_train = np.array(y_train)
    
    joblib.dump(y_train, Y_TRAIN_FILE)
    print(f"Saved y_train.joblib (Shape: {y_train.shape})")

    # --- 5. PROCESS TESTING LABELS ---
    # We must label the TEST set using the SAME K-Means model
    print("Loading K-Means model and X_test matrix...")
    kmeans_model = joblib.load(KMEANS_MODEL_FILE)
    X_test = joblib.load(os.path.join(FEATURE_DATA_DIR, "X_test.joblib"))

    print("Predicting clusters for X_test...")
    # Use the TRAINED model to predict which cluster each test cell belongs to
    test_cluster_assignments = kmeans_model.predict(X_test)
    
    # Loop and convert to 0s and 1s
    y_test = [create_label_from_cluster(cid, wildlife_map) for cid in test_cluster_assignments]
    y_test = np.array(y_test)
    
    joblib.dump(y_test, Y_TEST_FILE)
    print(f"Saved y_test.joblib (Shape: {y_test.shape})")

    # --- 6. (VISUAL) Show Label Distribution ---
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    train_bg = train_counts[0]
    train_wl = train_counts[1] if len(train_counts) > 1 else 0
    test_bg = test_counts[0]
    test_wl = test_counts[1] if len(test_counts) > 1 else 0

    print("\n--- Labeling Results ---")
    print(f"Training Set: {train_wl} 'Wildlife' cells ({train_wl/len(y_train):.2%}) | {train_bg} 'Background' cells")
    print(f"Testing Set:  {test_wl} 'Wildlife' cells ({test_wl/len(y_test):.2%}) | {test_bg} 'Background' cells")

    # Plot and save the distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Final Label Distribution')
    
    ax1.pie([train_bg, train_wl], labels=['Background (0)', 'Wildlife (1)'], 
            autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'], startangle=90)
    ax1.set_title(f'Training Set (n={len(y_train)})')
    
    ax2.pie([test_bg, test_wl], labels=['Background (0)', 'Wildlife (1)'], 
            autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'], startangle=90)
    ax2.set_title(f'Test Set (n={len(y_test)})')
    
    chart_path = os.path.join(OUTPUT_DIR, "final_label_distribution.png")
    plt.savefig(chart_path)
    print(f"\nSaved final label distribution chart to: {chart_path}")
    
    print("\n--- Step 4 Complete! ---")
    print("You now have X_train, y_train, X_test, and y_test.")
    print("You are ready to train the final model.")