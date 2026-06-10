# Palmprint Recognition for Authentication using Self-Supervised Learning

## Project Goal and Approach


This project implements a palmprint recognition system for authentication purposes. Due to the lack of labeled data for training, a self-supervised learning approach was adopted to learn meaningful feature representations from a large dataset of unlabeled palmprint images. The system utilizes contrastive learning to train a deep neural network to extract discriminative features, enabling palmprint matching and similarity comparison.  Finally, a user-friendly web interface was built to demonstrate the system's functionality.

## Dataset Description
For this project, a dataset of **12,000 unlabeled palmprint images** was used. The images are in **TIFF format** and were **obtained from the "Kaggle Palm Recognition Dataset for Authentication System"** dataset available on Kaggle ([https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data](https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data)).

This dataset is described as containing high-quality images designed for training and evaluating palm recognition models for biometric authentication.  While the original dataset includes categories for palm detection, palm vein patterns, and palm print matching, this project utilizes the **palmprint images suitable for palm print matching**. The dataset consists of images with **relatively high quality, designed for real-world biometric authentication applications.** The images are expected to exhibit variations inherent in biometric data collection, but are generally intended to be high-quality for effective model training. Each image in the dataset is also associated with metadata within the original Kaggle dataset, though this project focuses on the image data itself for self-supervised learning.

*   **Number of images:** 12,000
*   **Type of images:** Palmprint images
*   **Labeled or Unlabeled:** Unlabeled data
*   **Image format:** TIFF images
*   **Source:**  [Palm Recognition Dataset for Authentication System from Kaggle](https://www.kaggle.com/datasets/saqibshoaibdz/palm-dataset/data)
*   **Characteristics:** Images typically feature a dark background, which aids in segmenting the palm region from the surroundings. The dataset likely contains multiple images of palmprints from the same individual, reflecting a realistic biometric authentication scenario and hinting at potential for future identity-based analysis.While designed to be high quality, the dataset likely includes natural variations in palm appearance due to individual differences, minor pose variations, and subtle changes in skin texture.

## Image Preprocessing

Before feature extraction, palmprint images undergo a preprocessing pipeline to enhance relevant features and standardize the input for the model. The key stages are:

1.  **Grayscale Conversion:** The input color palmprint image (if any) is converted to grayscale to simplify processing and focus on texture and line patterns.
2.  **Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization):**  CLAHE is applied to improve the local contrast of the palmprint image, making the lines and ridges more prominent and easier to detect. This uses a clip limit of `2.0` and a tile grid size of `(5, 5)`.
3.  **Noise Reduction (Median Blur):** Median Blur with a kernel size of `1` is used to reduce noise while preserving edges.  While a kernel size of `1` effectively means minimal blurring, this step is included for potential future noise reduction adjustments.
4.  **Palm Segmentation (Thresholding and Contour Extraction):**  A thresholding technique (using a `threshold_value` of `80`) is applied to create a binary mask, separating the palm region (foreground) from the background.  Contours are then extracted to identify the palm's outline.
5.  **Region of Interest (ROI) Extraction (Centroid-Based):**  The Region of Interest (ROI) is extracted by finding the centroid (center of mass) of the largest contour (assumed to be the palm). A fixed-size square ROI of `(276x276)` pixels, centered at this centroid, is then cropped from the contrast-enhanced and noise-reduced grayscale image.
6.  **ROI Resizing:** The extracted ROI is resized to a standardized `(138x138)` pixel size to ensure consistent input dimensions for the feature extraction model.
7.  **ROI Normalization:** Finally, the pixel values of the resized ROI are normalized to the range of `[0, 1]` by dividing by `255.0`. This step helps in stabilizing and accelerating the training of the deep learning model.

### **Example Preprocessing Stages:**

Below are example images illustrating the key stages of the preprocessing pipeline applied to a sample palmprint image from the dataset, displayed in a horizontal flow:

<table>
  <tr>
    <td align="center">
      (1) Original Image:<br>
      <img src="output/original_image.png" width="150"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (2) Grayscale Conversion:<br>
      <img src="output/Grayscale%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (3) Contrast Enhanced (CLAHE):<br>
      <img src="output/CLAHE%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (4) Noise Reduction: <br>
      <img src="output/Blur%20Image_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (5) Contour Extraction: <br>
      <img src="output/Contours_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (6) (ROI) Extraction: <br>
      <img src="output/ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (7) ROI Resizing: <br>
      <img src="output/Resized%20ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (8) ROI Normalization: <br>
      <img src="output/Normalized%20ROI_screenshot_27.01.2025.png"><br>
    </td>
    <td align="center"> → </td>  <!-- Arrow -->
    <td align="center">
      (9) Final Preprocess Result: <br>
      <img src="output/Final%20Preprocess%20Image_screenshot_27.01.2025.png"><br>
    </td>
  </tr>
</table>


---
## Self-Supervised Learning Method Used

This project employed a **contrastive learning** approach, specifically inspired by **SimCLR (Simple Contrastive Learning for Representations)**.  Since the dataset is unlabeled, self-supervised learning was crucial to learn meaningful feature representations.

The SimCLR-inspired method works by:
1.  **Creating positive pairs:** For each palmprint image in a batch, two augmented versions are generated using various image transformations (see "Data Augmentation" section). These two augmented views of the same palmprint form a "positive pair."
2.  **Creating negative pairs:** Augmented views from different original palmprint images within the same batch are treated as "negative pairs."
3.  **Training objective:** The model is trained to learn feature representations (embeddings) such that:
    *   Embeddings of positive pairs are pulled closer together in the feature space.
    *   Embeddings of negative pairs are pushed further apart.

This contrastive objective forces the model to learn features that are invariant to the applied augmentations and discriminative between different palmprints, all without requiring explicit labels.

**Loss Function:** The training process utilizes the **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss** function to achieve this objective. This loss function, with a **temperature parameter of 0.07**, encourages similar embeddings for positive pairs and dissimilar embeddings for negative pairs in a normalized embedding space. By minimizing this loss, the model learns to extract robust and discriminative features from the unlabeled palmprint images.

**Data Augmentations:**
- `RandomResizedCrop`
- `RandomRotation` (±15°)
- `ColorJitter` (brightness & contrast, p=0.8)
- `GaussianBlur` (p=0.5)

## Model Architecture

<table>
  <tr>
    <td>
      <p align="center">
        <a href="output/model/7.%20palmprint_encoder.onnx.svg">
          <img src="output/model/7.%20palmprint_encoder.onnx.svg" height="500">
        </a>
</p>
    </td>
    <td>
      <p>
The feature extraction model is based on a <strong>ResNet-18</strong> Convolutional Neural Network (CNN) architecture.  The <strong>encoder</strong> part of the model utilizes a <strong>ResNet-18 backbone</strong>, initialized with weights <strong>pre-trained on ImageNet</strong>.  The original classification head of the pre-trained ResNet-18 was removed, and a <strong>2-layer Multilayer Perceptron (MLP) projection head</strong> was added on top of the encoder's output features.
        <br><br>
Specifically, the projection head consists of two linear layers with a ReLU activation function in between. This projection head further processes the features extracted by the ResNet-18 encoder before outputting the final <strong>256-dimensional feature embeddings</strong>.
        <br><br>
This architecture was chosen to leverage the powerful feature extraction capabilities of the ResNet-18 architecture, which is pre-trained on a large image dataset (ImageNet). The projection head is used to learn effective representations suitable for the contrastive learning task, mapping the ResNet-18 features into a lower-dimensional embedding space optimized for similarity comparisons. The model is trained to output <strong>256-dimensional feature embeddings</strong> for each input palmprint image.
      </p>
    </td>
  </tr>
</table>


## Training Process

The self-supervised model was trained using PyTorch. The training process involved:

*   **Framework:** PyTorch
*   **Optimizer:** Adam optimizer with a learning rate of `1e-4` (0.0001)
*   **Loss Function:** NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss with a temperature of `0.07`
*   **Training Epochs:** 100 epochs
*   **Batch Size:** 64
*   **Data Augmentations:** During training, the following data augmentations were applied to generate positive pairs:
    *   `RandomResizedCrop`
    *   `RandomRotation` (small angles)
    *   `ColorJitter` (grayscale - brightness and contrast adjustments)
    *   `GaussianBlur`
*   **Training Procedure:** The training loss was monitored, and the model weights were updated using backpropagation with the Adam optimizer to minimize the NT-Xent loss. The model learned to create embeddings where augmented views of the same palmprint are close together, and views of different palmprints are far apart in the embedding space.

## UI Functionality

*   **Framework used:** Streamlit
*   **Key features:** Image upload, display of uploaded image, display of top matching palmprints from database, similarity scores, indication of match/no match.

A simple and interactive web user interface was developed using **Streamlit** to demonstrate the palmprint recognition system. The UI provides the following functionalities:

*   **Image Upload:** Users can upload a palmprint image file (JPG, JPEG, PNG, TIFF).
*   **Uploaded Image Display:** The uploaded palmprint image is displayed on the UI.
*   **Matching Results Display:**  Upon uploading an image, the system compares it against a database of 11996 images pre-extracted palmprint features.
*   **Top Matching Palmprints:** The UI displays the top matches of 5 most similar palmprint images from the database, along with their cosine similarity scores.
*   **Match Indication:** The UI indicates whether a "match" is found based on a predefined similarity threshold value of 80%.

The UI allows users to easily test the palmprint recognition system by uploading their own palmprint images and observing the matching results in real-time.

![palmprint-demo.gif](output/palmprint-demo.gif)

_The demo showcases the palmprint recognition system's ability to identify palmprints from the same individual.  By using one of the original palmprint images (that was excluded from the training dataset), the model successfully located the other corresponding palmprint images in the database (from the same person) with similarity scores consistently around 90%._

### 🔍 Recognition Tab
- Upload a palmprint image (JPG, JPEG, PNG, TIFF)
- System preprocesses the image, extracts a 256-d embedding, and compares it against the full 12,000-image database via cosine similarity
- Displays the **top 5 matches** with similarity scores and match/no-match verdict at the 0.8 threshold

### 📊 Benchmark Tab
- Automatically builds **genuine pairs** (same identity, different images) and **impostor pairs** (different identities) from the embedding database
- Computes FAR, FRR, Valid Accuracy, and EER across the full dataset
- Provides an **interactive threshold slider** — metric cards update in real time without rerunning the expensive scoring step
- Renders a **FAR / FRR vs. Threshold curve** with annotated EER and operating-point markers
- Renders a **Score Distribution histogram** showing the separation between genuine and impostor similarity scores

![palmprint-demo.gif](output/palmprint-demo.gif)

---

## Benchmark Metrics

The benchmark tab evaluates the system using four standard biometric performance metrics.

### Metric Definitions

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Valid Accuracy** | (TA / total genuine) × 100 | % of genuine attempts the system correctly accepts |
| **FAR** (False Acceptance Rate) | (FA / total impostor) × 100 | % of impostor attempts the system incorrectly accepts |
| **FRR** (False Rejection Rate) | (FR / total genuine) × 100 | % of genuine attempts the system incorrectly rejects |
| **EER** (Equal Error Rate) | (FAR + FRR) / 2 at crossover | Threshold-independent summary — lower is better |

*TA = True Acceptances, FA = False Acceptances, FR = False Rejections*

### How to Interpret Results

**FAR / FRR trade-off** — raising the threshold makes the system stricter: FAR falls (fewer impostors accepted) but FRR rises (more genuine users rejected). Lowering the threshold has the opposite effect. The EER is the single-number summary of where these two curves cross.

**Score Distribution** — a well-trained model produces a bimodal histogram: genuine pairs cluster near **1.0** (high similarity) and impostor pairs cluster near **0.0** (low similarity). A large overlap indicates the model is struggling to discriminate identities.

**Operating point** — the default threshold of **0.8** is a conservative choice (prioritising low FAR). For scenarios that penalise false rejections more, lower the threshold towards the EER point.

---

## How to Run the Code

### Prerequisites

Ensure all steps below have been completed before launching the app.

```
project/
├── dataset/
│   └── preprocessed_images/     ← produced by step 2
├── output/
│   ├── model/
│   │   └── palmprint_encoder.pth  ← produced by step 3
│   └── embeddings/
│       ├── all_embeddings.npy     ← produced by step 4
│       └── image_names.npy        ← produced by step 4
```

### Step-by-step

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Preprocess images**
```bash
python preprocess/batch_preprocess.py
```
Reads `dataset/archive/session{1,2}/` and writes `.npy` files to `dataset/preprocessed_images/`.

**3. Train the encoder** *(skip if `palmprint_encoder.pth` already exists)*
```bash
cd self_supervised
python train.py
```
Saves `output/model/palmprint_encoder.pth` after 100 epochs.

**4. Extract and save embeddings**
```bash
python client/extract_features.py
```
Writes `output/embeddings/all_embeddings.npy` and `image_names.npy`.
The benchmark tab **requires these files** — run this step before opening the app.

**5. Launch the app**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Demo Walkthrough — Benchmark Tab

Once the app is running, navigate to the **📊 Benchmark** tab. The first load computes genuine / impostor pair scores over the full embedding database; subsequent threshold changes use the cached scores and respond instantly.

### What to explore

**Step 1 — Read the metric cards at the default threshold (0.8)**

At a high threshold the system is strict. Expect:
- **Low FAR** — very few impostors are mistakenly accepted
- **Higher FRR** — some genuine users are rejected because their score falls below 0.8
- **Valid Accuracy** — reflects how reliably the model recognises enrolled users

**Step 2 — Drag the threshold slider downward (towards 0.5)**

Watch how the metric cards change in real time:
- FAR rises as the system becomes more permissive
- FRR falls as more genuine pairs now exceed the threshold
- The orange operating-point marker on the FAR/FRR curve tracks your slider

**Step 3 — Locate the EER**

The grey dashed line on the curve marks the **EER threshold** — the point where FAR ≈ FRR. Moving the slider to this value minimises the total error rate and gives a balanced operating point. The purple EER card always shows this threshold-independent value regardless of where the slider sits.

**Step 4 — Inspect the Score Distribution**

The histogram separates the system's output into two groups:
- 🟢 **Green (genuine pairs)** — pairs from the same person; scores should be high
- 🔴 **Red (impostor pairs)** — pairs from different people; scores should be low

A large gap between the two peaks (small overlap) indicates strong identity discrimination. The orange vertical line shows where your selected threshold cuts across the distribution.

**Step 5 — Check the Evaluation Set Summary**

The counts panel shows the raw TA / FA / FR / TR numbers, making it straightforward to verify the metric formulas by hand.

---

## Key Libraries Used

| Library | Role |
|---------|------|
| **PyTorch** | Model building and training |
| **torchvision** | Pre-trained ResNet-18, image transforms |
| **OpenCV (cv2)** | Preprocessing pipeline |
| **NumPy** | Array operations, vectorised pair scoring |
| **scikit-learn** | Cosine similarity (recognition tab) |
| **Streamlit** | Web UI (both tabs) |
| **Matplotlib** | FAR/FRR curve and score distribution plots |
| **PIL (Pillow)** | Image handling |
