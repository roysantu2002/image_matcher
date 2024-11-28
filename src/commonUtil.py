import os
import random
import hashlib
import time
from datetime import datetime
import shutil
import json
import cv2
import numpy as np
from datetime import date

class CommonUtil:

    @staticmethod
    def generate_random_file(image1, image2, keyword=None):
        """
        Generates a unique file name.
        
        Args:
        - image1 (str): The first image name (or related identifier).
        - image2 (str): The second image name (or related identifier).
        - keyword (str, optional)
        
        Returns:
        - str: The path to the generated file.
        """
        try:
            # Create a base folder name using image1, image2, and keyword (if provided)
            base_name = f"{image1}_{image2}_{keyword}" if keyword else f"{image1}_{image2}"
            client_name_underscored = "_".join(base_name.split())
            
            # Create the folder path
            folder_path_underscored = os.path.join('media', client_name_underscored)

            # Ensure the directory exists
            if not os.path.exists(folder_path_underscored):
                os.makedirs(folder_path_underscored)

            # Create a unique identifier 
            current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            random_number = random.randint(1000, 9999)  # Generate a random number
            unique_identifier = hashlib.sha256(
                f"{client_name_underscored}_{current_date}_{random_number}".encode()).hexdigest()

            # Generate the file name using the keyword and unique identifier
            file_name = f"{keyword}_{unique_identifier}.png" if keyword else f"{unique_identifier}.png"
            file_path = os.path.join(folder_path_underscored, file_name)

            # Clean up old files in the folder that are older than 5 hours
            current_time = time.time()
            for file_ in os.listdir(folder_path_underscored):
                file_path_in_folder = os.path.join(folder_path_underscored, file_)
                try:
                    if os.path.isfile(file_path_in_folder) and (current_time - os.path.getmtime(file_path_in_folder)) > (5 * 3600):
                        os.remove(file_path_in_folder)
                        print(f"Deleted file: {file_path_in_folder}")
                except Exception as e:
                    print(f"Error deleting file: {file_path_in_folder}\n{e}")

            return file_path
        except Exception as e:
            print(f"Error generating random file: {e}")
            return None

    @staticmethod
    def create_image_folder(image1, image2, keyword):
        """
        Creates a folder.

        Args:
        - image1 (str)
        - image2 (str)
        - keyword (str)

        Returns:
        - str: The path to the created image folder.
        """
        try:
     
            base_name = f"{image1}_{image2}_{keyword}"
            name_underscored = "_".join(base_name.split())
            
            # Create the folder path
            folder_path_underscored = os.path.join('media', name_underscored)

            # Ensure the directory exists
            if not os.path.exists(folder_path_underscored):
                os.makedirs(folder_path_underscored)

            return folder_path_underscored
        except Exception as e:
            print(f"Error creating image folder: {e}")
            return None

    @staticmethod
    def extract_date_component(date_str, component):
        """
        Extracts a specific component ('dd', 'mm', 'yyyy') 
        
        Args:
            date_str (str): Date string in 'mm-dd-yyyy' format.
            component (str): The component to extract ('dd', 'mm', 'yyyy').
        
        Returns:
            int: The extracted component as an integer.
        """
        parts = date_str.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid date format: '{date_str}'. Expected format is 'mm-dd-yyyy'.")
        
        try:
            month, day, year = map(int, parts)
        except ValueError:
            raise ValueError(f"Invalid numeric values in date string: '{date_str}'. Expected integers in 'mm-dd-yyyy'.")

        try:
            date(year, month, day)
        except ValueError as e:
            raise ValueError(f"Invalid date: {e}. Check day, month, or year.")

        if component == "dd":
            return day
        elif component == "mm":
            return month
        elif component == "yyyy":
            return year
        else:
            raise ValueError(f"Invalid component: '{component}'. Expected 'dd', 'mm', or 'yyyy'.")

    @staticmethod
    def get_details_from_json(file_path, number):
        """
        Helper function to retrieve details for a specific number from a JSON file.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get(str(number), f"No details found {number}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return f"Error loading details: {str(e)}"

    @staticmethod
    def convert_first_letter_to_uppercase(lst):
        """
        """
        return [value.capitalize() for value in lst]

    @staticmethod
    def validate_image(image_path, valid_extensions=None, min_size=(100, 100), max_size=(5000, 5000)):
        """
        Validates the image by checking if the file.

        Args:
        image_path (str): Path to the image to be validated.
        valid_extensions (list)
        min_size (tuple)
        max_size (tuple)
        """
        # Check file existence
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        
        # Check file extension
        _, ext = os.path.splitext(image_path)
        valid_extensions = valid_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        if ext.lower() not in valid_extensions:
            raise ValueError(f"Invalid file: {ext}. Supported extensions: {valid_extensions}")
        
        # Check image size
        image = cv2.imread(image_path)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            raise ValueError(f"Image size {image.shape[:2]} is smaller than the minimum size {min_size}.")
        if image.shape[0] > max_size[1] or image.shape[1] > max_size[0]:
            raise ValueError(f"Image size {image.shape[:2]} exceeds the maximum size {max_size}.")
    
    def create_test_image(self, image_path, object_type="rectangle"):
        """
        Helper function to create a dummy image with different objects.
        Args:
        - image_path: Path where the image will be saved.
        - object_type: Type of object to draw in the image ('rectangle', 'circle', 'line', 'text').
        """
        # Create a black image (200x200 pixels)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        # Draw objects inside the image
        if object_type == "rectangle":
            # Draw a random rectangle (x, y, width, height, color, thickness)
            top_left = (random.randint(0, 100), random.randint(0, 100))
            bottom_right = (random.randint(101, 200), random.randint(101, 200))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            cv2.rectangle(image, top_left, bottom_right, color, -1)  # -1 fills the rectangle

        elif object_type == "circle":
            # Draw a random circle (center_x, center_y, radius, color, thickness)
            center = (random.randint(50, 150), random.randint(50, 150))
            radius = random.randint(20, 50)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            cv2.circle(image, center, radius, color, -1)  # -1 fills the circle

        elif object_type == "line":
            # Draw a random line (start_x, start_y, end_x, end_y, color, thickness)
            start_point = (random.randint(0, 200), random.randint(0, 200))
            end_point = (random.randint(0, 200), random.randint(0, 200))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            thickness = random.randint(1, 5)
            cv2.line(image, start_point, end_point, color, thickness)

        elif object_type == "text":
            # Write random text (at random position, font, size, color)
            text = "Test"
            position = (random.randint(10, 150), random.randint(50, 150))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = random.uniform(0.5, 2)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
            thickness = random.randint(1, 3)
            cv2.putText(image, text, position, font, font_scale, color, thickness)

        # Save the image to the specified path
        cv2.imwrite(image_path, image)

    def create_test_images(self, image_path, object_type="rectangle", object_count=3):
        """
        Creates a test image with multiple objects.

        Args:
        - image_path (str): Path to save the image.
        - object_type (str): Shape type ('rectangle', 'circle').
        - object_count (int): Number of objects to create.
        """
        # Create a blank white image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Add shapes
        for _ in range(object_count):
            x = np.random.randint(50, 350)
            y = np.random.randint(50, 350)
            size = np.random.randint(20, 100)
            if object_type == "rectangle":
                cv2.rectangle(img, (x, y), (x+size, y+size), (0, 0, 255), 3)  # Red rectangle
            elif object_type == "circle":
                cv2.circle(img, (x, y), size//2, (0, 255, 0), 3)  # Green circle
        
        # Save the image
        cv2.imwrite(image_path, img)

