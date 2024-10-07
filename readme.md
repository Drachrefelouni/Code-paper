
## Overview
This project performs pose localization by segmenting images using **HRNet** for semantic segmentation and **CASENet** for edge detection. The P3P algorithm is then used to match 2D image keypoints with 3D points from the Dubrovnik dataset.

## Prerequisites
Ensure you have the following software installed on your machine:
- **Python 3.x**
- **pip** (Python package manager)

## Required Python Packages
You need to install the following packages:
- **OpenCV** (`cv2`)
- **PyTorch** (with CUDA support if using a GPU)
- **NumPy**
- **Pillow** (Python Imaging Library)

You can install these packages by running:
```bash
pip install opencv-python 
pip install torch torchvision 
pip install numpy pillow
pip install matplotlib scikit-image
```

## Instructions

### Step 1: Segment the Image Using HRNet
1. Download a pre-trained **HRNet** model for semantic segmentation.
2. Prepare your input image (e.g., `image.jpg`).
3. Run the HRNet segmentation script to obtain a segmented output.

### Step 2: Segment Edges Using CASENet
1. Download a pre-trained **CASENet** model for edge detection.
2. Use the segmented image from the previous step as input to the CASENet edge detection script.
3. Save the edge detection output for further processing.

### Step 3: Download the Dubrovnik Dataset
Download the Dubrovnik dataset from the following link:
- [Dubrovnik Dataset](http://s3.amazonaws.com/LocationRecognition/Datasets/Dubrovnik6K.tar.gz)

1. Extract the downloaded dataset to access the 3D points and descriptors.

### Step 4: Execute the Pose Localization Code
1. Ensure you have the segmented image and edge map ready.
2. Place the extracted Dubrovnik dataset in the appropriate directory for the pose localization script.
3. Run the pose localization script to perform 2D-3D matching using the P3P algorithm. Specify the input image, object 3D points, and camera matrix as needed.

## Example Command
After setting everything up, you can execute the pose localization script from the terminal:
```bash
python pose_localization.py --image_path path_to_your_image --dataset_path path_to_dubrovnik_dataset
```


## References for Models
- **HRNet (High-Resolution Network)**: 
  - DOI: [10.1109/CVPR42600.2023.00114](https://doi.org/10.1109/CVPR42600.2020.00114)
  - Reference: Sun, K., Wang, D., & Liu, J. (2023). High-Resolution Representations for Labeling Pixels and Regions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

- **CASENet (Category-Aware Semantic Edge Detection)**: 
  - DOI: [10.1109/CVPR.2019.159](https://doi.org/10.1109/CVPR.2017.159)
  - Reference: Yu, Z., Liu, H., & Ramalingam, S. (2019). CASENet: Deep Category-Aware Semantic Edge Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).




