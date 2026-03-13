🖼️ **Computer Vision Lab Tasks (OpenCV – Jupyter Notebook)**



This repository contains multiple Computer Vision mini-projects implemented using Python, OpenCV, NumPy, Matplotlib, and Scikit-Image in Jupyter Notebook.



📌 **Initial Image Processing Operations**



✔ Basic Image Handling



Load image using OpenCV



Convert BGR → RGB for visualization



Convert to Grayscale



Resize image



Display using Matplotlib subplots



Save processed images



🚀 **Implemented Features**



1️⃣ **Image Quality Enhancer (Social Media Ready)**



Automatic brightness \& contrast adjustment



Histogram-based clipping



Resize to 1080×1080 (Instagram format)



Side-by-side comparison (Original vs Enhanced)



2️⃣ **Security Camera Grayscale Archiver**



Convert colored image to grayscale



Save grayscale version for storage efficiency



Visual comparison display



3️⃣ **Image Comparison for Duplicates**



Resize images to uniform size



Convert to grayscale



Compute:



MSE (Mean Squared Error)



SSIM (Structural Similarity Index)



Display comparison results



4️⃣ **Color Filter for Object Detection**



Convert image to HSV color space



Apply red color mask using HSV range



Detect and highlight red objects



Show original vs detected output



5️⃣ **License Plate Blurring (Privacy Protection)**



Define ROI (Region of Interest)



Apply Gaussian Blur



Replace region in original image



Save protected image



🆕 Additional Implemented Tasks



6️⃣ **Custom Image Watermarking Tool**



Load main image + PNG watermark



Supports watermark with or without transparency



Adjustable opacity



Centered watermark placement



Proportional resizing based on image size



Save final watermarked output



7️⃣ **Basic OCR Preprocessing Pipeline**



Load scanned document



Convert to grayscale



Apply noise removal (Gaussian / Median filtering)



Apply thresholding for clean text extraction



Prepare image for OCR engines



8️⃣ **2×2 Photo Collage Generator**



Upload 4 images



Resize to uniform dimensions



Arrange into 2 rows × 2 columns grid



Save final collage image



9️⃣ **Batch Color-to-Grayscale Converter**



Read all images from directory



Convert each to grayscale



Save with \_bw suffix



Designed for print shop black-and-white workflows



🔟 **Noise Removal in Scanned Documents**



Read noisy scanned image



Apply Gaussian Blur or Median Filter



Apply thresholding



Improve readability for documentation



🔄 **Navigation Flow**



Each task follows the same structure:



Load Image



Preprocess (if required)



Apply Core Operation



Display Results (Matplotlib Subplots)



Save Output using cv2.imwrite()



Print Confirmation Message



This consistent workflow ensures clarity, readability, and easy debugging.



🛠️ **Technologies Used**



Python 3



OpenCV (cv2)



NumPy



Matplotlib



Scikit-Image (for SSIM)



📂 **Output Files**



Processed images are saved locally in the specified directory paths:



Grayscale images



Resized images



Enhanced images



Blurred license plate image



Watermarked images



Collage image



OCR processed image



**🎯 Purpose**



This project demonstrates practical applications of:



Image enhancement



Image transformation



Object detection



Privacy protection



Similarity measurement



Batch processing



OCR preprocessing



Watermarking

