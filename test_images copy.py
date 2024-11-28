import unittest
import os
import shutil
import json
import cv2
import numpy as np
from commonUtil import CommonUtil
from imageMatcher import ImageMatcher

common_obj = CommonUtil()

image_path = 'test_image_rectangle.jpg'
        
# Create an image with a rectangle
# common_obj.create_test_image(image_path, object_type="rectangle")
# common_obj.validate_image(image_path)

# Create image1 with rectangles
common_obj.create_test_images("image1.jpg", object_type="rectangle", object_count=5)
# Create image2 with a subset and different arrangement of rectangles
common_obj.create_test_images("image2.jpg", object_type="rectangle", object_count=3)

image_matcher = ImageMatcher("image1.jpg", "image2.jpg")

# Match objects between the two images
matches, annotated_image = image_matcher.match_objects()

# Save and display the results
image_matcher.save_and_display_results(matches, annotated_image)