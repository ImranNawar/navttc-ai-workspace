# Real-time Face Recognition using Haar Cascade & SIFT  

## Overview  
This project implements real-time face recognition using **Haar Cascade** for face detection and **SIFT (Scale-Invariant Feature Transform)** for feature extraction and matching.  

## Steps  

1. **Dataset Collection:** Gather face images of classmates.  
2. **Face Detection:** Use Haar Cascade to detect faces in images.  
   - Download the Haar Cascade XML: [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades).  
3. **Feature Extraction:** Detect keypoints using SIFT and store them in `features.npy`.  
4. **Real-time Recognition:**  
   - Capture frames from the webcam.  
   - Detect faces using Haar Cascade.  
   - Extract keypoints with SIFT.  
   - Match keypoints with stored features using **BF Matcher**.  
5. **Face Identification:** Display the detected face with the corresponding name in real time.  

## Usage  

1. Run all cells in `code.ipynb` to generate `features.npy` and `labels.npy`.  
2. Execute the last cell to start the webcam-based face recognition system.  

### Performance  
- **FPS:** 20 - 30  