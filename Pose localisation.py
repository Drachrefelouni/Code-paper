import numpy as np
import cv2
import torch

# Method to run P3P algorithm using OpenCV
def estimate_pose_p3p(image_points, object_points, camera_matrix):
    # Ensure we have at least 4 points for P3P
    if len(image_points) >= 4:
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_P3P)
        if success:
            return rvec, tvec
        else:
            print("P3P failed to estimate pose")
            return None, None
    else:
        print("Not enough points for P3P")
        return None, None

# Keypoint Labeling Method 1: SoftMax Labeling
def softmax_labeling(softmax_output, threshold):
    labels = []
    for prob in softmax_output:
        max_prob = np.max(prob)
        if max_prob > threshold:
            labels.append(np.argmax(prob))
        else:
            labels.append(np.where(prob > threshold)[0])
    return labels

# Keypoint Labeling Method 2: Semantic Segmentation with Edge Detection
def semantic_edge_labeling(softmax_output, edge_map, threshold):
    labels = []
    for i, prob in enumerate(softmax_output):
        max_prob = np.max(prob)
        if edge_map[i]:
            labels.append(np.where(prob > threshold)[0])  # Multiple labels if on the edge
        else:
            labels.append(np.argmax(prob))  # Single label for non-edge pixels
    return labels

# Keypoint Labeling Method 3: Combination of SoftMax and Semantic Edge
def combined_labeling(semantic_softmax, edge_softmax, threshold):
    labels = []
    for sem_prob, edge_prob in zip(semantic_softmax, edge_softmax):
        combined_prob = (sem_prob + edge_prob) / 2
        max_prob = np.max(combined_prob)
        if max_prob > threshold:
            labels.append(np.argmax(combined_prob))
        else:
            labels.append(np.where(combined_prob > threshold)[0])
    return labels

# Function to match 2D image keypoints with 3D points
def match_2d_3d(image_keypoints, descriptors, object_points, object_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors, object_descriptors)
    matched_image_points = []
    matched_object_points = []
    
    for match in matches:
        matched_image_points.append(image_keypoints[match.queryIdx].pt)
        matched_object_points.append(object_points[match.trainIdx])
        
    return np.array(matched_image_points), np.array(matched_object_points)

# Main function to run the full pipeline
def main(image_path, object_3d_points, object_3d_descriptors, camera_matrix, method=1):
    # Load query image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use SIFT to extract keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Match 2D keypoints with 3D points
    matched_image_points, matched_object_points = match_2d_3d(keypoints, descriptors, object_3d_points, object_3d_descriptors)

    # Apply one of the three labeling methods
    if method == 1:
        # Apply SoftMax Labeling
        softmax_output = probabilities("data.npg")
        labels = softmax_labeling(softmax_output, threshold=0.3)
    elif method == 2:
        # Apply Semantic Segmentation with Edge Detection
        softmax_output = probabilities("data.npg")
        edge_map = extract_edge_map("example_edge.png")
        labels = semantic_edge_labeling(softmax_output, edge_map, threshold=0.3)
    elif method == 3:
        # Apply Combination of SoftMax and Semantic Edge Detection
        softmax_output = probabilities("data.npg")
        edge_map = extract_edge_map("example_edge.png")
        labels = combined_labeling(semantic_softmax, edge_softmax, threshold=0.3)
    else:
        raise ValueError("Invalid method selection!")

    # Estimate pose using P3P
    rvec, tvec = estimate_pose_p3p(matched_image_points, matched_object_points, camera_matrix)

    if rvec is not None and tvec is not None:
        print(f"Rotation Vector: {rvec}")
        print(f"Translation Vector: {tvec}")
    else:
        print("Pose estimation failed.")

# Example usage
if __name__ == "__main__":
    #  camera matrix (intrinsics)
    camera_matrix = np.array([[718.856, 0, 607.1928],
                              [0, 718.856, 185.2157],
                              [0, 0, 1]])


    object_3d_points = load_dubrovnik_3D_points('3D_points.3D') # Random 3D points as an example
    object_3d_descriptors = np.random.rand(1000, 128)  # Random descriptors as an example


    main("query_image.jpg", object_3d_points, object_3d_descriptors, camera_matrix, method=1)

