import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class ImageMatcher:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def detect_features(self, image_path):
        """Detect keypoints and descriptors using SIFT (or SURF if needed)"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use SIFT for better feature detection (can be replaced with SURF if desired)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """Match descriptors between two images using FLANN"""
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=10), {}  # FLANN with KDTrees
        )
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        return good_matches

    def annotate_image(self, image, matches, keypoints1, keypoints2, labels):
        """Annotate Image 2 with corresponding labels from Image 1"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()

        # Create a dictionary to store objects found in Image 2
        detected_objects = {}

        for match in matches:
            img2_idx = match.trainIdx
            img1_idx = match.queryIdx

            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt

            # Assign a label to Image 2 (cyclically if there are fewer labels than keypoints)
            label = labels[img1_idx % len(labels)]

            # Annotate Image 2 with labels
            draw.text((x2, y2), label, font=font, fill=(255, 0, 0))

            # Record the label of the object detected in Image 2
            detected_objects[label] = (x2, y2)

        # Print the objects found in Image 2
        print("Objects detected in Image 2:")
        for label, coords in detected_objects.items():
            print(f"Object: {label} at location {coords}")

        return pil_image, detected_objects

    def match_objects(self):
        """Match objects in Image 1 to Image 2 and annotate Image 2"""
        keypoints1, descriptors1 = self.detect_features(self.image1_path)
        keypoints2, descriptors2 = self.detect_features(self.image2_path)

        # Ensure there are keypoints in Image 1
        if not keypoints1:
            print("No keypoints detected in Image 1")
            return None, None

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
        annotated_image, detected_objects = self.annotate_image(image2, matches, keypoints1, keypoints2, labels)

        result_image_path = 'annotated_image.jpg'
        self.save_and_display_results(matches, annotated_image, result_image_path)

        return matches, annotated_image, detected_objects  # Return matches, annotated image, and detected objects

    def save_and_display_results(self, matches, annotated_image, result_image_path):
        """Save and display the annotated image"""
        annotated_image.save(result_image_path)  # Save the annotated image
        print(f"Annotated image saved to {result_image_path}")

        # Display the annotated image
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()