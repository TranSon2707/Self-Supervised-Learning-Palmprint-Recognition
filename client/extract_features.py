import os
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from self_supervised.model import PalmprintEncoder

PREPROCESSED_DATA_DIR = 'dataset/preprocessed_images'

ENCODER_PATH = 'output/model/palmprint_encoder.pth'
encoder = PalmprintEncoder().encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.load_state_dict(torch.load(ENCODER_PATH, weights_only=False, map_location=device))
encoder.eval()
encoder.to(device)


image_files = [f for f in os.listdir(PREPROCESSED_DATA_DIR) if f.endswith('.npy')]
all_embeddings = []
image_names = []

for image_file in image_files:
    image_path = os.path.join(PREPROCESSED_DATA_DIR, image_file)

    try:
        preprocessed_roi = np.load(image_path)
        preprocessed_roi = np.expand_dims(preprocessed_roi, axis=0)
        preprocessed_roi = torch.from_numpy(preprocessed_roi).float()
        preprocessed_roi = preprocessed_roi.repeat(1, 3, 1, 1).to(device)


        with torch.no_grad():
            embedding = encoder(preprocessed_roi).cpu().numpy()

        all_embeddings.append(embedding.flatten())
        image_names.append(os.path.splitext(image_file)[0])


    except Exception as e:
        print(f"Error processing {image_file}: {e}")

all_embeddings = np.array(all_embeddings)

# ensure output embeddings directory exists and use a project-relative path
embeddings_dir = os.path.join('output', 'embeddings')
os.makedirs(embeddings_dir, exist_ok=True)

np.save(os.path.join(embeddings_dir, 'all_embeddings.npy'), all_embeddings)
np.save(os.path.join(embeddings_dir, 'image_names.npy'), image_names)

print(f"Feature extraction complete. Embeddings saved to {os.path.join(embeddings_dir, 'all_embeddings.npy')}, image names saved to {os.path.join(embeddings_dir, 'image_names.npy')}")