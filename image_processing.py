from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "processed_images"
VISUAL_OUTPUT_FOLDER = "preprocessing_visuals" # For our new charts
EXAMPLE_FOLDER = os.path.join(VISUAL_OUTPUT_FOLDER, "example_images")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EXAMPLE_FOLDER, exist_ok=True)

LOG_FILE = "preprocessing_log.txt"

# --- Constants ---
TARGET_RATIO = 4.0 / 3.0
TARGET_SIZE = (800, 600)
RATIO_TOLERANCE = 0.01
NUM_EXAMPLES = 5 # Save 5 before/after examples

def center_crop_to_4by3(img):
    width, height = img.size
    current_ratio = width / float(height)
    if abs(current_ratio - TARGET_RATIO) < RATIO_TOLERANCE:
        return img
    if current_ratio > TARGET_RATIO:
        new_width = int(height * TARGET_RATIO)
        left = (width - new_width) // 2
        right = left + new_width
        top, bottom = 0, height
    else:
        new_height = int(width / TARGET_RATIO)
        top = (height - new_height) // 2
        bottom = top + new_height
        left, right = 0, width
    
    return img.crop((left, top, right, bottom))

# --- Main Script ---
if __name__ == "__main__":
    log_path = os.path.join(OUTPUT_FOLDER, LOG_FILE)
    print(f"Starting processing...")
    
    processed_count = 0
    discarded_count = 0
    error_count = 0
    examples_saved = 0
    
    # --- NEW: List to store original sizes for histogram ---
    original_widths = []

    with open(log_path, "w") as log:
        log.write("--- Image Processing Log ---\n")
        all_files = os.listdir(INPUT_FOLDER)
        
        for filename in all_files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            file_path = os.path.join(INPUT_FOLDER, filename)
            
            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    original_widths.append(img.width) # Store original width
                    
                    # --- NEW: Save "before" example ---
                    save_example = (examples_saved < NUM_EXAMPLES)
                    if save_example:
                        img.save(os.path.join(EXAMPLE_FOLDER, f"{filename}_0_original.jpg"))

                    # --- Rule 1a: CROP images to 4:3 ---
                    img_4by3 = center_crop_to_4by3(img)
                    width, height = img_4by3.size
                    
                    if save_example and img_4by3 != img: # Save if it was cropped
                         img_4by3.save(os.path.join(EXAMPLE_FOLDER, f"{filename}_1_cropped_4x3.jpg"))

                    # --- Apply Rules 1b and 1c ---
                    if width == 800 and height == 600:
                        img_4by3.save(os.path.join(OUTPUT_FOLDER, filename))
                        processed_count += 1
                        if save_example: # Already 800x600, just copy
                            img_4by3.save(os.path.join(EXAMPLE_FOLDER, f"{filename}_2_final.jpg"))
                            examples_saved += 1
                            
                    elif width > 800:
                        # Rule 1b: Scale DOWN
                        final_image = img_4by3.resize(TARGET_SIZE, Image.LANCZOS)
                        final_image.save(os.path.join(OUTPUT_FOLDER, filename))
                        processed_count += 1
                        if save_example:
                            final_image.save(os.path.join(EXAMPLE_FOLDER, f"{filename}_2_final_resized.jpg"))
                            examples_saved += 1
                    else:
                        # Rule 1c: Discard (too small)
                        log.write(f"DISCARDED (Too Small): {filename} (Final 4:3 size {width}x{height})\n")
                        discarded_count += 1
            except Exception as e:
                log.write(f"ERROR: {filename} ({e})\n")
                error_count += 1

    print("\n--- Step 1 Complete! ---")
    print(f"Processed and Saved: {processed_count}")
    print(f"Discarded (Too Small): {discarded_count}")

    # --- NEW VISUAL 1: Histogram of Original Image Widths ---
    plt.figure(figsize=(10, 6))
    plt.hist(original_widths, bins=50, edgecolor='black')
    plt.title('Distribution of Original Image Widths (Before Processing)')
    plt.xlabel('Image Width (pixels)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    hist_path = os.path.join(VISUAL_OUTPUT_FOLDER, "original_widths_histogram.png")
    plt.savefig(hist_path)
    print(f"Saved width histogram to: {hist_path}")
    plt.close()

    # --- NEW VISUAL 2: Pie Chart of Processed vs. Discarded ---
    labels = 'Processed', 'Discarded (Too Small)'
    sizes = [processed_count, discarded_count]
    colors = ['#4CAF50', '#FF5252'] # Green, Red
    if processed_count > 0 or discarded_count > 0:
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('Image Processing Results')
        plt.axis('equal') # Ensures pie is a circle
        pie_path = os.path.join(VISUAL_OUTPUT_FOLDER, "processing_pie_chart.png")
        plt.savefig(pie_path)
        print(f"Saved pie chart to: {pie_path}")
        plt.close()