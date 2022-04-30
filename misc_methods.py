""" 
Name: misc_methods.py

Author: Ryan Hood
Email: ryanchristopherhood@gmail.com

Description: This file is part of a larger project which performs a perspective 
transformation on images of checks.  This file contaisn misc methods that are used by other files.
"""

import cv2


def determine_landscape(unordered_corners):
    """ This method takes a list of corners in no particular ordering and determines 
    if the check is portrait or landscape. """
    
    # First, let's find the biggest horizontal and vertical distances.
    biggest_horizontal_distance = 0
    biggest_vertical_distance = 0
    for first_tuple_index in range(0, len(unordered_corners)):
        first_tup = unordered_corners[first_tuple_index]
        for second_tuple_index in range(0, len(unordered_corners)):
            second_tup = unordered_corners[second_tuple_index]
            
            x1 = first_tup[0]
            y1 = first_tup[1]
            x2 = second_tup[0]
            y2 = second_tup[1]
            
            horizontal_distance = abs(x2 - x1)
            vertical_distance = abs(y2 - y1)
            
            if horizontal_distance > biggest_horizontal_distance:
                biggest_horizontal_distance = horizontal_distance
            if vertical_distance > biggest_vertical_distance:
                biggest_vertical_distance = vertical_distance
            
        
    is_landscape = 0
    
    if biggest_vertical_distance < biggest_horizontal_distance:
        is_landscape = 1
        
    return is_landscape







    

def rotate_image(img):
    """ This method checks the current orientation of image.  If it is portrait, make it landscape.
    If it is landscape, leave it as is. """ 
    rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image
      

def make_check_upright(img):
    """ This method takes an image of a check which may or may not be upside-down and 
    it returns the check right-side-up. """

    # The strategy is to first apply thresholding, so that writing is white and everything else is black.
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY) 

    # Cut Check
    cut = cut_check(threshold)

    # Now, we analyze the top and bottom of the check for the first black pixel.
    count_from_top, count_from_bottom = count_rows_until_black(cut)
       
    if count_from_top < count_from_bottom:
        # Then we need to flip the check.
        img = cv2.rotate(img, cv2.ROTATE_180)
        print("CHECK BELIEVED TO BE UPSIDE DOWN ... SO FLIPPED")
    
    return img

def count_rows_until_black(img):
    """  """
    img_height = img.shape[0]
        
    # First, let's focus on top.
    count_from_top = 0
    while True:
        # Analyze current row.
        row = img[count_from_top][:]
        
        if 0 in row:
            break
        
        count_from_top = count_from_top + 1
        
    # Next, let's focus on bottom.
    count_from_bottom = 0
    while True:
        # Analyze current row.
        row = img[img_height - 1 - count_from_bottom][:]
        
        if 0 in row:
            break
        
        count_from_bottom = count_from_bottom + 1
    
    return count_from_top, count_from_bottom


def cut_check(img):
    """  """
    # First, let's resize to uniform size.
    img = cv2.resize(img, (1500, 700), interpolation = cv2.INTER_AREA)
    
    # Cut the sides off.
    img = img[25:675, 300:1200]
    
    return img
