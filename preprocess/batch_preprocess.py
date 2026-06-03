import os
import numpy as np

# from preprocess.preprocessor import preprocess_image
from preprocessor import preprocess_image

"""
Takes original palmprint images from the dataset/archive, applies preprocessing from `preprocessor.py`
saves the preprocessed ROIs as .npy files in a new directory (`dataset/preprocessed_images`) for efficient loading during training in `train.py`.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

original_images_dirs = {
    "session1": os.path.join(BASE_DIR, "dataset", "archive", "session1"),
    "session2": os.path.join(BASE_DIR, "dataset", "archive", "session2")
}
preprocessed_images_dir = os.path.join(BASE_DIR, "dataset", "preprocessed_images")
os.makedirs(preprocessed_images_dir, exist_ok=True)



def batch_preprocess():
    for session_name, original_images_dir in original_images_dirs.items():
        image_files = [f for f in os.listdir(original_images_dir) if        # '../dataset/archive/session1'
                       f.lower().endswith(('.tiff', '.tif'))]

        for image_file in image_files:
            image_path = os.path.join(original_images_dir, image_file)
            preprocessed_roi = preprocess_image(image_path)

            if preprocessed_roi is not None:
                unique_file_name = f"{session_name}_{image_file.replace('.tiff', '.npy').replace('.tif', '.npy')}"
                save_path = os.path.join(preprocessed_images_dir, unique_file_name)
                np.save(save_path, preprocessed_roi)
                print(f"Preprocessed and saved: {image_file} -> {save_path}")
            else:
                print(f"Preprocessing failed for: {image_file}")

    print("Preprocessing complete!")


batch_preprocess()
