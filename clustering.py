import os
import joblib
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
NUM_CLUSTERS = 60       # The K value you are testing
NUM_SAMPLES_TO_SHOW = 30
K_RANGE_FOR_ELBOW = range(10, 51, 5) # Test K from 10 to 50 (steps of 5)

FEATURE_DATA_DIR = "feature_data"
X_TRAIN_FILE = os.path.join(FEATURE_DATA_DIR, "X_train_unlabeled.joblib")
META_TRAIN_FILE = os.path.join(FEATURE_DATA_DIR, "meta_train.joblib")
PROCESSED_IMAGES_DIR = "processed_images"

OUTPUT_DIR = "clustering_output"
VISUALS_DIR = os.path.join(OUTPUT_DIR, "cluster_visuals") # For cell images
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts") # For our new plots
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

KMEANS_MODEL_FILE = os.path.join(OUTPUT_DIR, "kmeans_model.joblib")
ASSIGNMENTS_FILE = os.path.join(OUTPUT_DIR, "cluster_assignments.joblib")

CELL_W = 100
CELL_H = 75

# --- 2. LOAD FEATURE MATRIX ---
print(f"Loading feature matrix: {X_TRAIN_FILE}")
try:
    X_train = joblib.load(X_TRAIN_FILE)
    meta_train = joblib.load(META_TRAIN_FILE)
except FileNotFoundError:
    print("Error: Could not find feature data. Did Step 2 run?")
    exit()

print(f"Loaded {X_train.shape[0]} cell features.")

# --- NEW VISUAL 1: The Elbow Method (to justify K) ---
# This part can take a long time, but you only need to run it once
# to find your 'k'. You can comment it out after.
print("Running Elbow Method to find optimal K...")
inertias = []
for k in tqdm(K_RANGE_FOR_ELBOW):
    # Use MiniBatchKMeans for speed, with a subset of data
    subset_indices = np.random.choice(X_train.shape[0], 20000, replace=False)
    X_subset = X_train[subset_indices]
    
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans_elbow.fit(X_subset)
    inertias.append(kmeans_elbow.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_RANGE_FOR_ELBOW, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
elbow_path = os.path.join(CHARTS_DIR, "elbow_method_plot.png")
plt.savefig(elbow_path)
print(f"Saved Elbow Method plot to: {elbow_path}")
plt.close()

# --- 3. RUN K-MEANS CLUSTERING (Main) ---
print(f"Running K-Means with k={NUM_CLUSTERS} (this may take a while)...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=5)
kmeans.fit(X_train)
joblib.dump(kmeans, KMEANS_MODEL_FILE)
print("K-Means model saved.")

# --- 4. GET CLUSTER ASSIGNMENTS ---
print("Getting cluster assignments for all cells...")
cluster_assignments = kmeans.labels_
joblib.dump(cluster_assignments, ASSIGNMENTS_FILE)

# --- NEW VISUAL 2: Cluster Size Distribution ---
cluster_counts = np.bincount(cluster_assignments, minlength=NUM_CLUSTERS)
plt.figure(figsize=(12, 7))
plt.bar(range(NUM_CLUSTERS), cluster_counts, edgecolor='black')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Cells')
plt.title('Cluster Size Distribution (How many cells in each cluster)')
plt.xticks(range(NUM_CLUSTERS))
plt.grid(axis='y', alpha=0.75)
dist_path = os.path.join(CHARTS_DIR, "cluster_distribution_chart.png")
plt.savefig(dist_path)
print(f"Saved cluster distribution chart to: {dist_path}")
plt.close()

# --- 5. VISUALIZE THE CLUSTERS (Same as before) ---
print(f"Saving {NUM_SAMPLES_TO_SHOW} sample images for each cluster...")
cluster_paths = {}
for i in range(NUM_CLUSTERS):
    cluster_folder = os.path.join(VISUALS_DIR, f"cluster_{i}")
    os.makedirs(cluster_folder, exist_ok=True)
    cluster_paths[i] = cluster_folder

cluster_counts_saved = {i: 0 for i in range(NUM_CLUSTERS)}

for (filename, r, c), cluster_id in tqdm(zip(meta_train, cluster_assignments), total=len(meta_train)):
    if cluster_counts_saved[cluster_id] >= NUM_SAMPLES_TO_SHOW:
        continue
    try:
        img_path = os.path.join(PROCESSED_IMAGES_DIR, filename)
        image = cv2.imread(img_path)
        y1, y2 = r * CELL_H, (r + 1) * CELL_H
        x1, x2 = c * CELL_W, (c + 1) * CELL_W
        cell_image = image[y1:y2, x1:x2]
        sample_path = os.path.join(cluster_paths[cluster_id], f"{filename}_r{r}_c{c}.jpg")
        cv2.imwrite(sample_path, cell_image)
        cluster_counts_saved[cluster_id] += 1
    except Exception as e:
        print(f"\nWarning: Error processing {filename}: {e}")

print("\n--- Step 3 Complete! ---")
print(f"All charts saved to: {CHARTS_DIR}")
print(f"All sample cell images saved in: {VISUALS_DIR}")
print("\nNEXT STEP: Manually review the folders in 'cluster_visuals'.")