import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV

# --- 1. CONFIGURATION ---

# --- Input Folders ---
FEATURE_DATA_DIR = "feature_data"
LABEL_DATA_DIR = "final_data"

# --- Output Folder ---
MODEL_DIR = "final_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Files ---
X_TRAIN_FILE = os.path.join(FEATURE_DATA_DIR, "X_train_unlabeled.joblib")
Y_TRAIN_FILE = os.path.join(LABEL_DATA_DIR, "y_train.joblib")
X_TEST_FILE = os.path.join(FEATURE_DATA_DIR, "X_test.joblib")
Y_TEST_FILE = os.path.join(LABEL_DATA_DIR, "y_test.joblib")

FINAL_MODEL_FILE = os.path.join(MODEL_DIR, "wildlife_model.joblib")

# --- 2. (OPTIONAL) HYPERPARAMETER TUNING ---
# Set this to 'True' to run GridSearchCV to find the BEST model.
# It is VERY SLOW but gives higher accuracy.
# Set to 'False' to run a single, fast, good-enough model.
PERFORM_GRID_SEARCH = False 

# --- 3. LOAD ALL DATA ---
print("Loading all training and testing data...")
try:
    X_train = joblib.load(X_TRAIN_FILE)
    y_train = joblib.load(Y_TRAIN_FILE)
    X_test = joblib.load(X_TEST_FILE)
    y_test = joblib.load(Y_TEST_FILE)
except FileNotFoundError:
    print("Error: Could not find all data files. Did all previous steps run?")
    exit()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 4. TRAIN THE MODEL ---

if PERFORM_GRID_SEARCH:
    print("\nStarting GridSearchCV (this will take a long time)...")
    # These are parameters to test
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # We use 'class_weight="balanced"' to handle the imbalanced data
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    
    # We focus on 'f1' score because accuracy is useless here
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='f1', verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model from the search
    model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
else:
    print("\nTraining a single Random Forest model...")
    # We use 'class_weight="balanced"' to automatically handle
    # the fact that we have way more 0s than 1s. This is critical.
    model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   class_weight="balanced", n_jobs=-1)
    
    model.fit(X_train, y_train)

print("Model training complete.")

# --- 5. SAVE THE FINAL MODEL ---
joblib.dump(model, FINAL_MODEL_FILE)
print(f"Final trained model saved to: {FINAL_MODEL_FILE}")

# --- 6. EVALUATE ON TEST SET ---
print("\n--- Model Evaluation on TEST Data ---")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for '1'

# --- (VISUAL 1) Classification Report ---
# This is the most important text output for your presentation
report = classification_report(y_test, y_pred, target_names=['Background (0)', 'Wildlife (1)'])
print(report)
# Save to file
with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
    f.write("\nNOTE: Focus on the 'Wildlife (1)' row, especially F1-Score.\n")

# --- (VISUAL 2) Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"Saved confusion matrix to: {cm_path}")
plt.close()


# --- (VISUAL 3) ROC Curve and AUC Score ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_path = os.path.join(MODEL_DIR, "roc_curve.png")
plt.savefig(roc_path)
print(f"Saved ROC curve to: {roc_path}")
plt.close()


print("\n--- Step 5 Complete! ---")