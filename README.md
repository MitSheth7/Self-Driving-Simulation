# Self-Driving Simulation

This project implements a self-driving car simulation using deep learning techniques, focusing on training a model to predict steering angles based on images captured from the car's camera in a simulated environment. The main goal is to develop a Convolutional Neural Network (CNN) that can interpret visual data and make driving decisions in real-time, emulating human driving behavior.

## Overview

The simulation uses a dataset comprising images of the road alongside corresponding steering angles, which are organized in a CSV file. The process begins with data collection, followed by preprocessing, augmentation, and ultimately training the model. Here’s a brief rundown of how it works:

1. **Data Collection**: Images from the car's perspective are stored alongside their respective steering angles.

2. **Data Preprocessing**:
   - **Cropping**: Images are cropped to focus on the area of interest, removing unnecessary parts of the image.
   - **Color Space Conversion**: The images are converted from RGB to YUV, enhancing the model's ability to process them.
   - **Gaussian Blur**: A Gaussian blur is applied to reduce noise in the images.
   - **Resizing**: Images are resized to a uniform shape (200x66 pixels) for consistent input to the model.
   - **Normalization**: Pixel values are scaled to the range [0, 1] by dividing by 255.

3. **Data Augmentation**: Techniques such as random panning, zooming, and brightness adjustments are used to artificially increase the dataset's diversity, allowing the model to generalize better.

4. **Model Architecture**: The CNN consists of several convolutional layers followed by dense layers. The architecture is designed to learn and extract important features from the images to accurately predict steering angles.

5. **Training**: The model is trained using mean squared error (MSE) loss, optimizing the weights based on the difference between predicted and actual steering angles.

6. **Real-Time Prediction**: Once trained, the model can process live images from the car's camera, predict the steering angle, and send control commands to steer the vehicle.
   
![image](https://github.com/user-attachments/assets/d1ce7df4-0c87-4f5e-9817-e0fa21330159)

![image](https://github.com/user-attachments/assets/6b3b1f5e-5883-4d1d-b6ad-55fbba3313b4)


![left_2024_10_04_18_12_39_951](https://github.com/user-attachments/assets/2ed0507f-121c-4986-8815-9617411e67f5)

