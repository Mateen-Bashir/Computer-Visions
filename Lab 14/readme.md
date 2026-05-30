\# Handwritten Alphabet Recognition using CNN



\## Project Overview



This project implements a Convolutional Neural Network (CNN) for handwritten alphabet recognition using the EMNIST Letters dataset. The trained model is capable of recognizing handwritten English alphabets and is further applied to multiple real-world character recognition scenarios.



\## Dataset



\- Dataset: EMNIST Letters

\- Total Classes: 26 (A-Z)

\- Image Size: 28 × 28 pixels

\- Format: CSV files



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- OpenCV

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-Learn



\## Tasks Performed



\### 1. Data Preprocessing



\- Loaded EMNIST training and testing datasets

\- Merged datasets

\- Removed invalid labels

\- Reshaped images into 28×28 format

\- Corrected image orientation

\- Normalized pixel values

\- Visualized class distribution

\- Displayed random handwritten samples



\### 2. CNN Model Development



\- Convolution Layers

\- Max Pooling Layers

\- Batch Normalization

\- Dense Layers

\- Dropout Regularization

\- Softmax Classification



\### 3. Model Training



\- Train-Test Split

\- Sparse Categorical Crossentropy

\- SGD Optimizer

\- Accuracy Monitoring

\- Validation Tracking



\### 4. Model Evaluation



\- Test Accuracy Calculation

\- Test Loss Calculation

\- Accuracy Graph Visualization

\- Loss Graph Visualization

\- Random Sample Predictions



\### 5. Custom Image Recognition



\- Image Loading

\- Noise Reduction

\- Adaptive Thresholding

\- Contour Detection

\- Character Segmentation

\- Character Prediction



\## Real-World Applications Implemented



\### CAPTCHA Recognition

Recognizes distorted alphabetical characters from CAPTCHA images.



\### License Plate Recognition

Extracts and predicts characters from vehicle license plates.



\### Postal Automation

Recognizes handwritten characters from postal addresses.



\### Smart Device Handwriting Recognition

Converts handwritten notes into editable text.



\### Historical Manuscript Digitization

Recognizes faded and ancient handwritten characters from manuscript images.



\## Model Output



The model predicts:



A, B, C, D, E ... Z



from handwritten character images.



\## Future Enhancements



\- Multi-language support

\- CRNN Architecture

\- Transformer Models

\- Real-time Mobile Deployment

\- Cloud-Based OCR Services



\## Author



Mateen Bashir

