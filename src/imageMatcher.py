import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ImageMatcher:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def detect_features(self, image_path):
        """Detect ORB features in an image and return keypoints and descriptors"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """Match features between two images"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def annotate_image(self, image, matches, keypoints1, keypoints2, labels):
        """Annotate Image 2 with corresponding labels from Image 1"""
        # Convert image from OpenCV format to PIL format for annotation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()  # Using default font

        # Loop through the matches and annotate with labels
        for match in matches:
            # Get the position of matched keypoints
            img2_idx = match.trainIdx
            img1_idx = match.queryIdx

            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt

            # Example: Annotate Image 2 with the label from Image 1
            label = labels[img1_idx]  # Assuming `labels` is a list of labels for Image 1
            draw.text((x2, y2), label, font=font, fill=(255, 0, 0))

        return pil_image

    def save_and_display_results(self, matches, annotated_image, result_image_path):
        """Save the annotated image"""
        annotated_image.save(result_image_path)
        annotated_image.show()
        
    def match_objects(self):
        """Match objects in Image 1 to Image 2 and annotate Image 2"""
        keypoints1, descriptors1 = self.detect_features(self.image1_path)
        keypoints2, descriptors2 = self.detect_features(self.image2_path)

        # Ensure there are keypoints in Image 1
        if not keypoints1:
            print("No keypoints detected in Image 1")
            return None, None  # Return None if no keypoints in Image 1

        # Dynamically create labels based on the number of keypoints in Image 1
        labels = [f"Object{i+1}" for i in range(len(keypoints1))]

        # Ensure keypoints and descriptors are valid before matching
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            print("Descriptors are empty. Cannot proceed with matching.")
            return None, None

        matches = self.match_features(keypoints1, descriptors1, keypoints2, descriptors2)

        if not matches:
            print("No good matches found.")
            return None, None

        image2 = cv2.imread(self.image2_path)
        annotated_image = self.annotate_image(image2, matches, keypoints1, keypoints2, labels)

        result_image_path = 'annotated_image.jpg'
        self.save_and_display_results(matches, annotated_image, result_image_path)

        return matches, annotated_image  # Return the matches and annotated imag