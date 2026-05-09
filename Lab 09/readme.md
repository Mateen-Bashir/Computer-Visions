# Content-Based Image Retrieval (CBIR) Lab Tasks

## Overview
This project implements Content-Based Image Retrieval (CBIR) systems using classical computer vision techniques. The system retrieves visually similar images based on feature extraction and similarity measurement.

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Feature Extraction Methods
- HSV Color Histogram
- Grayscale Histogram
- Texture-based descriptors (basic histogram-based approximation)

---

## Case Studies

### 1. Fashion Retail Image Search
- Query: Clothing image
- Dataset: DeepFashion / fashion images
- Features: HSV color histogram

### 2. Medical Image Retrieval
- Query: MRI / X-ray image
- Dataset: Medical scans
- Features: Grayscale histogram

### 3. Wildlife Image Retrieval
- Query: Animal image (e.g., tiger)
- Dataset: Wildlife images
- Features: HSV color histogram

### 4. Art Recommendation System
- Query: Painting image
- Dataset: Art images (WikiArt)
- Features: HSV histogram

### 5. Industrial Defect Detection
- Query: Product surface image
- Dataset: Defect-free/defective samples
- Features: Grayscale histogram

---

## Output
- Query image is compared with dataset images
- Top-3 most similar images are displayed
- Similarity is computed using cosine similarity

---

## How to Run
1. Place dataset in folder
2. Provide query image
3. Run notebook step by step

---

## Conclusion
CBIR systems effectively retrieve visually similar images using feature-based comparison without requiring deep learning models.