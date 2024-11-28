import unittest
import os
import shutil
import json
import cv2
import numpy as np
from src.commonUtil import CommonUtil
from src.imageMatcher import imageMatcher

class TestCommonUtil(unittest.TestCase):

    def test_validate_image_with_rectangle(self):
        image_path = 'test_image_rectangle.jpg'
        
        # Create an image with a rectangle
        CommonUtil.create_test_image(image_path, object_type="rectangle")
        
        try:
            # Now validate the image
            CommonUtil.validate_image(image_path)
        finally:
            # Clean up the test image after the test
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_validate_image_with_circle(self):
        image_path = 'test_image_circle.jpg'
        
        # Create an image with a circle
        CommonUtil.create_test_image(image_path, object_type="circle")
        
        try:
            # Now validate the image
            CommonUtil.validate_image(image_path)
        finally:
            # Clean up the test image after the test
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_validate_image_valid(self):
        # Assume we have a valid image file at 'test_image.jpg'
        image_path = 'test_image.jpg'

        # Create a simple dummy image to save as test
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(image_path, image)
        
        try:
            # Validate the image (assuming it's a valid image)
            CommonUtil.validate_image(image_path)
        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_validate_image_with_text(self):
        image_path = 'test_image_text.jpg'
        
        # Create an image with text
        CommonUtil.create_test_image(image_path, object_type="text")
        
        try:
            # Now validate the image
            CommonUtil.validate_image(image_path)
        finally:
            # Clean up the test image after the test
            if os.path.exists(image_path):
                os.remove(image_path)

    def test_generate_random_file(self):
        # Test the random file generation function
        image1 = 'image1.jpg'
        image2 = 'image2.jpg'
        keyword = 'test'
        
        folder = CommonUtil.create_image_folder(image1, image2, keyword)
        random_file = CommonUtil.generate_random_file(image1, image2, keyword)
        
        self.assertTrue(random_file.endswith('.png'))
        self.assertTrue(os.path.exists(folder))
        
        # Cleanup
        if os.path.exists(random_file):
            os.remove(random_file)
        if os.path.exists(folder):
            shutil.rmtree(folder)

    def test_create_image_folder(self):
        # Create a folder and check if it exists
        image1 = 'image1.jpg'
        image2 = 'image2.jpg'
        keyword = 'test_folder'
        
        folder_path = CommonUtil.create_image_folder(image1, image2, keyword)
        
        self.assertTrue(os.path.exists(folder_path))
        
        # Cleanup
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    def test_extract_date_component(self):
        # Testing valid date extraction
        date_str = "12-25-2024"
        component = "mm"
        self.assertEqual(CommonUtil.extract_date_component(date_str, component), 12)

        # Testing invalid date component
        with self.assertRaises(ValueError):
            CommonUtil.extract_date_component(date_str, "month")
        
        # Testing invalid date format
        with self.assertRaises(ValueError):
            CommonUtil.extract_date_component("2024-25-12", "dd")

    def test_get_details_from_json(self):
        # Test the JSON details retrieval
        json_data = {
            "1": {"name": "Image1", "description": "Test image 1"},
            "2": {"name": "Image2", "description": "Test image 2"}
        }

        # Mock JSON file content
        mock_json_path = 'mock_data.json'
        with open(mock_json_path, 'w') as f:
            json.dump(json_data, f)

        result = CommonUtil.get_details_from_json(mock_json_path, 1)
        self.assertEqual(result, {"name": "Image1", "description": "Test image 1"})

        result = CommonUtil.get_details_from_json(mock_json_path, 3)
        self.assertEqual(result, "No details found 3")
        
        # Cleanup
        if os.path.exists(mock_json_path):
            os.remove(mock_json_path)


class TestImageObjectMatcher(unittest.TestCase):

    def test_image_object_matcher_valid(self):
        # Assume valid image paths
        image1_path = 'test_image1.jpg'
        image2_path = 'test_image2.jpg'

        # Create simple dummy images to test
        image1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image2 = np.zeros((200, 200, 3), dtype=np.uint8)

        cv2.imwrite(image1_path, image1)
        cv2.imwrite(image2_path, image2)
        
        # Instantiate ImageObjectMatcher
        matcher = imageMatcher(image1_path, image2_path)

        matches, annotated_image = matcher.match_objects()

        # Check if matches are found (in this case, dummy images, so expected to be empty)
        self.assertEqual(len(matches), 0)
        
        # Check that the annotated image is not None and is a valid image
        self.assertIsNotNone(annotated_image)
        self.assertEqual(annotated_image.shape, (200, 200, 3))  # Same size as the input images
        
        # Cleanup
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)

    def test_invalid_image_paths(self):
        # Test invalid image paths
        invalid_image1 = 'invalid_image1.jpg'
        invalid_image2 = 'invalid_image2.jpg'
        
        # Test for validation errors when files do not exist
        with self.assertRaises(FileNotFoundError):
            imageMatcher(invalid_image1, invalid_image2)

    def test_random_file_generation_in_matcher(self):
        # Test random file generation when saving the result
        image1_path = 'test_image1.jpg'
        image2_path = 'test_image2.jpg'

        # Create simple dummy images to test
        image1 = np.zeros((200, 200, 3), dtype=np.uint8)
        image2 = np.zeros((200, 200, 3), dtype=np.uint8)

        cv2.imwrite(image1_path, image1)
        cv2.imwrite(image2_path, image2)

        # Instantiate ImageObjectMatcher and match objects
        matcher = imageMatcher(image1_path, image2_path)
        matches, annotated_image = matcher.match_objects()

        # Test if the result is saved with a random filename
        random_filename = CommonUtil.generate_random_file(image1_path, image2_path, 'match_results')
        self.assertTrue(random_filename.endswith('.png'))

        # Cleanup
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)
        if os.path.exists(random_filename):
            os.remove(random_filename)


if __name__ == '__main__':
    unittest.main()
