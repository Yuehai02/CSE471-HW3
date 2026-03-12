# VisionEdge: Mobile vs Cloud YOLOv5 Inference

An end-to-end Android + Flask system for comparing **on-device** and **server-side** object detection using **YOLOv5**.

## Overview

VisionEdge is a lightweight computer vision project that evaluates two deployment strategies for real-time object detection:

- **Mobile Inference**: run YOLOv5 directly on an Android device
- **Cloud Inference**: send the image to a Flask server, run YOLOv5 remotely, and return detection results to the app

The project focuses on the tradeoff between **latency, efficiency, privacy, and deployment flexibility**. By testing multiple YOLOv5 model sizes under the same workflow, it highlights when mobile inference is practical and when cloud inference is the better choice.

## Key Features

- Android client for image-based object detection
- Flask backend for cloud inference
- Support for multiple YOLOv5 variants
- Side-by-side comparison of mobile and cloud execution
- Bounding box visualization on detection results
- Latency-oriented evaluation across different model sizes

## Supported Models

- YOLOv5n
- YOLOv5s
- YOLOv5m
- YOLOv5l
- YOLOv5x

## System Architecture

### Mobile Inference Pipeline

1. User selects or loads an image in the Android app
2. The app preprocesses the image locally
3. A YOLOv5 model runs directly on the device
4. Post-processing is applied to generate final detections
5. The app renders bounding boxes and displays results

### Cloud Inference Pipeline

1. User selects or loads an image in the Android app
2. The app sends the image to the Flask server
3. The server preprocesses the image
4. The server runs YOLOv5 inference
5. Detection results are returned to the Android app
6. The app visualizes the final output

## Tech Stack

### Mobile
- Android
- Java
- PyTorch Mobile / TorchScript

### Backend
- Python
- Flask
- PyTorch

### Model
- YOLOv5

## File Description

- `MainActivity.java`  
  Main Android activity responsible for user interaction, image loading, and inference workflow selection.

- `PrePostProcessor.java`  
  Handles image preprocessing and post-processing logic such as resizing, normalization, and detection parsing.

- `ResultView.java`  
  Renders detection results and bounding boxes on the Android side.

- `serve.py`  
  Flask server that receives images, performs cloud-side YOLOv5 inference, and returns prediction results.

- `yolo v5 n/`, `yolo v5 s/`, `yolo v5 m/`, `yolo v5 l/`, `yolo v5 x/`  
  Model folders for different YOLOv5 variants used in the comparison.

## Motivation

Deploying object detection models on edge devices introduces important engineering tradeoffs. Smaller models can run locally with better privacy and independence from network connectivity, while larger models often benefit from the stronger compute resources available on a server.

This project was built to explore those tradeoffs in a practical Android application and measure how model size affects the feasibility of mobile inference versus cloud inference.

## Experimental Focus

The main goal of this project is to compare:

- Inference latency
- Scalability across model sizes
- Practicality of deployment on mobile devices
- Advantages and limitations of cloud-based inference

## Main Takeaways

- **Cloud inference** is generally more suitable for heavier models and lower response time under stronger compute resources.
- **Mobile inference** is more self-contained and privacy-friendly, but becomes less efficient as model size increases.
- The choice between mobile and cloud deployment depends on the target use case, especially the balance between **speed, privacy, connectivity, and hardware constraints**.

## How to Run

### 1. Start the Flask Server

Make sure Python dependencies are installed, then run `python serve.py`.

### 2. Open the Android Project

- Open the project in Android Studio
- Build and run the app on an emulator or Android device
- Make sure the app is configured correctly to communicate with the Flask server for cloud inference

### 3. Run Detection

- Load an image in the app
- Choose the inference mode
- View detection boxes and compare performance between mobile and cloud execution

## Potential Improvements

- Add live camera support instead of image-only input
- Benchmark on real Android hardware instead of emulator-only testing
- Add automated latency logging and visualization
- Extend the backend to support batch inference
- Compare YOLOv5 with newer lightweight detection models
