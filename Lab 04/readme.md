**Image Enhancement Techniques using OpenCV**

This repository contains implementations of several image enhancement techniques using Python, OpenCV, NumPy, and Matplotlib.

These techniques improve image quality under challenging conditions such as low lighting, fog, underwater environments, and thermal imagery.



**Implemented Techniques**

1\. Contrast Stretching

Improves image contrast by expanding the range of intensity values.



Formula:



I\_new = ((I - I\_min) / (I\_max - I\_min)) \* 255



**Used for:**



Low contrast images

Foggy environments

Satellite images

2\. Gamma Correction

Gamma correction adjusts brightness using a nonlinear transformation.



Formula:



I\_out = 255 \* (I\_in / 255)^gamma



Effects:



γ < 1 → Brightens image

γ > 1 → Darkens image

3\. Histogram Equalization

Enhances global contrast by redistributing pixel intensity values.



**Applications:**



X-ray images

Low-light scenes

Satellite imagery

4\. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Improves local contrast by applying histogram equalization to small regions.



**Advantages:**



Reduces noise amplification

Improves visibility in night scenes

Case Studies Implemented

Case Study 1 – Underwater Image Enhancement

Enhancing underwater images using histogram equalization and gamma correction.



Case Study 2 – Nighttime Road Scene Enhancement

Using CLAHE to improve visibility for autonomous vehicles.



Case Study 3 – Security Camera Enhancement

Applying contrast stretching and gamma correction to brighten low-light surveillance footage.



Case Study 4 – Foggy Urban Environment Enhancement

Using contrast stretching and histogram equalization to improve visibility in traffic monitoring cameras.



Case Study 5 – Thermal Satellite Image Enhancement

Applying gamma correction and histogram equalization to highlight temperature gradients.



**Technologies Used**

Python

OpenCV

NumPy

Matplotlib

Applications

Computer Vision

Autonomous Vehicles

Smart City Surveillance

Environmental Monitoring

Marine Biology Research

