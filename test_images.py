import os
from src.commonUtil import CommonUtil
from src.imageMatcher import ImageMatcher

# Initialize the CommonUtil object
common_obj = CommonUtil()

# Define the base folder path for saving images (without nested subfolders)
base_folder = "media/test_images"  # Or any other base directory

# Ensure that the base folder exists
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# Define image names and keyword
image1 = 'image1'
image2 = 'image2'
keyword = 'test_images'

# Generate file paths directly under the base folder without creating new subfolders
image1_path = os.path.join(base_folder, f"{image1}.jpg")
image2_path = os.path.join(base_folder, f"{image2}.jpg")

# Create test images with rectangles and save them inside the base folder
common_obj.create_test_images(image1_path, object_type="rectangle", object_count=5)
common_obj.create_test_images(image2_path, object_type="rectangle", object_count=3)

# Create ImageMatcher object for comparing the two images
image_matcher = ImageMatcher(image1_path, image2_path)

# Match objects between the two images and handle potential None returns from match_objects()
result = image_matcher.match_objects()

if result:
    matches, annotated_image, detected_objects = result

    # Save and display the results inside the same base folder
    result_image_path = os.path.join(base_folder, "result_image.jpg")
    image_matcher.save_and_display_results(matches, annotated_image, result_image_path)

    print(f"Image matching results saved at {result_image_path}.")
    print(f"Detected objects in Image 2:")
    for label, coords in detected_objects.items():
        print(f"Object: {label} found at location {coords}")
else:
    print("Image matching failed. No objects matched between the images.")
