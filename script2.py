from PIL import Image
import os

# Folders
input_folder = "images"
output_folder = "output_images_cropped"
non_4by3_list = "output_images/non_4by3_images.txt"   # list of non-4:3 image filenames

os.makedirs(output_folder, exist_ok=True)

# Desired final ratio and size
TARGET_RATIO = 4 / 3
TARGET_SIZE = (800, 600)

def center_crop_to_4by3(img):
    """Crop the image to the largest possible 4:3 region, centered."""
    width, height = img.size
    current_ratio = width / height

    if current_ratio > TARGET_RATIO:
        # Image is too wide → crop horizontally
        new_width = int(height * TARGET_RATIO)
        left = (width - new_width) // 2
        right = left + new_width
        top, bottom = 0, height
    else:
        # Image is too tall → crop vertically
        new_height = int(width / TARGET_RATIO)
        top = (height - new_height) // 2
        bottom = top + new_height
        left, right = 0, width

    return img.crop((left, top, right, bottom))

with open(non_4by3_list, "r") as f:
    files = [line.strip() for line in f.readlines() if line.strip()]

for i, filename in enumerate(files, 1):
    file_path = os.path.join(input_folder, filename)
    if not os.path.exists(file_path):
        print(f"Skipping (not found): {filename}")
        continue
    try:
        with Image.open(file_path) as img:
            # Convert to RGB to avoid mode issues (e.g. PNG with alpha)
            img = img.convert("RGB")

            # Crop to center 4:3 region
            cropped = center_crop_to_4by3(img)

            # Resize to 800x600
            resized = cropped.resize(TARGET_SIZE, Image.LANCZOS)

            # Save to output folder
            output_path = os.path.join(output_folder, filename)
            resized.save(output_path)

            print(f"[{i}/{len(files)}] Cropped and resized: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\n Done! All non-4:3 images are now center-cropped and saved to:")
print(output_folder)
