# Code Implementation

This implementation demonstrates object matching and annotation between two images using feature matching. The goal is to detect objects in Image 2, match them to their corresponding objects in Image 1, and annotate Image 2 with the labels derived from Image 1.

## Source Code

```python
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ImageMatcher:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def detect_features(self, image_path):
        """Detect keypoints and descriptors in the image using ORB"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """Match descriptors between two images using FLANN"""
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
            {}
        )
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def annotate_image(self, image, matches, keypoints1, keypoints2, labels):
        """Annotate Image 2 with corresponding labels from Image 1"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()

        for match in matches:
            img2_idx = match.trainIdx
            img1_idx = match.queryIdx

            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt

            # Assign a label to Image 2 (cyclically if there are fewer labels than keypoints)
            label = labels[img1_idx % len(labels)]
            draw.text((x2, y2), label, font=font, fill=(255, 0, 0))

        return pil_image

    def match_objects(self):
        """Match objects in Image 1 to Image 2 and annotate Image 2"""
        keypoints1, descriptors1 = self.detect_features(self.image1_path)
        keypoints2, descriptors2 = self.detect_features(self.image2_path)

        # Dynamically create labels based on the number of keypoints in Image 1
        labels = [f"Object{i+1}" for i in range(len(keypoints1))]

        matches = self.match_features(keypoints1, descriptors1, keypoints2, descriptors2)

        image2 = cv2.imread(self.image2_path)
        annotated_image = self.annotate_image(image2, matches, keypoints1, keypoints2, labels)

        result_image_path = 'annotated_image.jpg'
        self.save_and_display_results(matches, annotated_image, result_image_path)

    def save_and_display_results(self, matches, annotated_image, result_image_path):
        """Save and display the annotated image"""
        annotated_image.save(result_image_path)
        cv2.imshow('Matched Image', cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```
