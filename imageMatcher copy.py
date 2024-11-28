import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from commonUtil import CommonUtil
from PIL import Image

class ImageMatcher:
    def __init__(self, image1_path, image2_path, min_size=(100, 100), max_size=(5000, 5000), valid_extensions=None):
        """
        Initializes the ImageMatcher with the given image paths, size constraints, and valid extensions.

        Args:
        - image1_path (str): Path to the first image.
        - image2_path (str): Path to the second image.
        - min_size (tuple): Minimum allowed size for images.
        - max_size (tuple): Maximum allowed size for images.
        - valid_extensions (list): List of valid image extensions.
        """
        self.image1_path = image1_path
        self.image2_path = image2_path
        self.min_size = min_size
        self.max_size = max_size
        self.valid_extensions = valid_extensions or ['.jpg', '.jpeg', '.png', '.bmp']

        # Validate images using CommonUtil
        CommonUtil.validate_image(image1_path, self.valid_extensions, self.min_size, self.max_size)
        CommonUtil.validate_image(image2_path, self.valid_extensions, self.min_size, self.max_size)

        self.image1 = cv2.imread(image1_path)
        self.image2 = cv2.imread(image2_path)

    def _extract_features(self, image):
        """
        Extracts keypoints and descriptors from the image using ORB.

        Args:
        - image (ndarray): Image array.

        Returns:
        - keypoints (list): List of keypoints detected in the image.
        - descriptors (ndarray): Descriptors corresponding to the keypoints.
        """
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def _match_features(self, descriptors1, descriptors2):
        """
        Matches descriptors between two images using a brute-force matcher.

        Args:
        - descriptors1 (ndarray): Descriptors from the first image.
        - descriptors2 (ndarray): Descriptors from the second image.

        Returns:
        - matches (list): List of matched keypoints.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def match_objects(self):
        """
        Matches objects between image1 and image2 based on feature points.

        Returns:
        - matches (list): List of matched keypoints.
        - annotated_image (ndarray): Annotated image with matched keypoints.
        """
        # Extract features from both images
        keypoints1, descriptors1 = self._extract_features(self.image1)
        keypoints2, descriptors2 = self._extract_features(self.image2)
        
        # Match features between the two images
        matches = self._match_features(descriptors1, descriptors2)

        # Annotate matched points on image2
        annotated_image = self.image2.copy()
        for match in matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt
            cv2.putText(annotated_image, f"({int(x1)},{int(y1)})", (int(x2), int(y2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return matches, annotated_image

    def save_and_display_results(self, matches, annotated_image, result_image_path):
        # Convert the numpy array to a PIL Image
        if isinstance(annotated_image, np.ndarray):  # Ensure it's a numpy array
            annotated_image = Image.fromarray(annotated_image)

        # Save the annotated image
        annotated_image.save(result_image_path)
        
        # Display the results (if necessary, like using matplotlib or OpenCV)
        print(f"Results saved at {result_image_path}")
        # If you want to display the image:
        annotated_image.show()
