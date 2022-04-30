""" 
Name: driver.py

Author: Ryan Hood
Email: ryanchristopherhood@gmail.com

Description: This file is the main driving script for the project.  A folder is provided 
by the user as input.  This folder should contain images of checks at various angles. 
A perspective transformation will be applied so that a new image is returned of the checks 
with a 'straight on' view.  For each image in the folder, the original image and the modified image 
is shown, but nothing is written.
"""

import argparse
import os
import cv2
import find_initial_corners as fic
import misc_methods as misc
import perspective_transform_methods as ptm

    

def process_img(img_path):
    """ This method processes a single image, and shows both the original image and the new image. """
    img_original = cv2.imread(img_path)
    
    print("Processing image located here: ", img_path)
    
    # STEP 1: Find the corners of the original check.  Also find the target corners.
    original_contour = fic.find_quadrilateral_contour(img_original)
    unordered_corners = fic.extract_four_corners_from_contour(original_contour)
    
    is_landscape = misc.determine_landscape(unordered_corners)
    
    source_corners = ptm.configure_source_corners(unordered_corners, is_landscape)
    target_corners = ptm.configure_resulting_corners(is_landscape)
    
    # STEP 2: Perform the prospective transform.
    corrected_image = ptm.perform_perspective_transform(img_original, source_corners, target_corners)
        
    # STEP 3: Clean up the resulting corrected image.    
    if is_landscape == 0:    
        cropped = corrected_image[20:1500,20:700]
        landscape_check = misc.rotate_image(cropped)
    else:
        cropped = corrected_image[20:700,20:1500]
        landscape_check = cropped
        
    upright_check = misc.make_check_upright(landscape_check)
    
    cv2.imshow("Original", img_original)
    cv2.imshow("Result", upright_check)
    cv2.waitKey(0)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str, default='samples', help='check images folder')
    
    args = parser.parse_args()
    input_folder = args.input_folder
   
    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png','.jpg','.jpeg', '.bmp', '.gif', '.tiff')):
            process_img(img_path)
            