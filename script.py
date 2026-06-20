from PIL import Image
import os

# Input and output directories
input_folder = "images"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# File to store names of non-4:3 images
log_file = os.path.join(output_folder, "non_4by3_images.txt")

# Helper function to check 4:3 ratio
def is_four_by_three(width, height, tolerance=0.01):
    ratio = width / height
    return abs(ratio - (4/3)) <= tolerance

with open(log_file, "w") as log:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(input_folder, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size

                    # Check if 4:3 aspect ratio
                    if is_four_by_three(width, height):
                        # Resize only if larger than 800x600
                        if width > 800 or height > 600:
                            img_resized = img.resize((800, 600), Image.LANCZOS)
                            img_resized.save(os.path.join(output_folder, filename))
                        else:
                            # Keep smaller image as-is
                            img.save(os.path.join(output_folder, filename))
                    else:
                        # Log non-4:3 images
                        log.write(filename + "\n")

            except Exception as e:
                print(f"Skipping {filename}: {e}")

print("Processing complete!")
print(f"Non-4:3 images logged in: {log_file}")
