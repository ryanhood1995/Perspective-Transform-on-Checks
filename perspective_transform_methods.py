""" 
Name: perspective_transform_methods.py

Author: Ryan Hood
Email: ryanchristopherhood@gmail.com

Description: This file is part of a larger project which performs a perspective 
transformation on images of checks.  This file in particular provides methods useful 
for setting up and performing perspective transformation.
"""

import numpy as np
import cv2

def max_sum_index(tuples):  
    """ This method finds the index of the max and min element of a list. """
    maxi = -1
    mini = -1
    max_val = 0
    min_val = 10000000
    # traversal in the lists 
    for i,x in enumerate(tuples): 
        sum = 0 
        # traversal in tuples in list 
        for y in x: 
            sum+= y
        
        if sum > max_val:
            max_val = sum
            maxi = i 
        if sum < min_val:
            min_val = sum
            mini = i
          
    return maxi, mini

def configure_source_corners(corners_list, is_landscape):
    """ This method takes an unordered list of corners and finds the correct order.  The order must be correct since 
    we will need to map them to the target corners. """
    
    # Point 1 is the point in the top-left corner of the image.  It will have the minimum value 
    # if the coordinates are added together.
    
    # Point 4 is the point in the bottom-right corner of the image.  It will have the maximum value 
    # if the coordinates are added together.
    
    # Point 2 is the point in the top-right corner of the image.  It, along with Point 3, will be found
    # after Point 1 and Point 4 are known.
    
    max_index, min_index = max_sum_index(corners_list)
    
    x1 = corners_list[min_index][0]
    y1 = corners_list[min_index][1]
    
    x4 = corners_list[max_index][0]
    y4 = corners_list[max_index][1]
    
    # Now, let's remove those corners from corners_list.
    corners_list = [v for i, v in enumerate(corners_list) if i not in [min_index, max_index]]
    
    # Now we analyze the two remaining elements to get which is P2 and which is P3.
    if corners_list[0][0] > corners_list[1][0] and corners_list[0][1] < corners_list[1][1]:
        x2 = corners_list[0][0]
        y2 = corners_list[0][1]
        x3 = corners_list[1][0]
        y3 = corners_list[1][1]
    else:
        x2 = corners_list[1][0]
        y2 = corners_list[1][1]
        x3 = corners_list[0][0]
        y3 = corners_list[0][1]
        
    
    source_corners = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    
    return source_corners
    

def configure_resulting_corners(is_landscape):
    """ This method takes corners from a check, and configures the corners that these will map to in the 
    resulting image after persepctive transforming. """

    # The configured corners will be different depending on if check is landscape or not.
    if is_landscape == 0:
        target_corners = np.float32([(20, 20), (700, 20), (20, 1500), (700, 1500)])
        
    else:
        target_corners = np.float32([(20, 20), (1500, 20), (20, 700), (1500, 700)])
        
    return target_corners


def perform_perspective_transform(img, input_corners, target_corners):
    """ This method takes an image with input and target corners and finds the transformation matrix to do a 
    perspective transformation and then applies that transformation matrix to an image."""
    h, w = img.shape[:2]
    
    # We use the input_corners and the target_corners to get the transformation matrix.
    M = cv2.getPerspectiveTransform(input_corners, target_corners)
    
    # Apply that transformation matrix on the image.
    warped = cv2.warpPerspective(img, M, (w, h), flags = cv2.INTER_LINEAR)
        
    return warped
