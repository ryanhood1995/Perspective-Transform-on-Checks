""" 
Name: hough_line_methods.py

Author: Ryan Hood
Email: ryanchristopherhood@gmail.com

Description: This file is part of a larger project which performs a perspective 
transformation on images of checks.  This file in particular provides methods useful 
for detecting the corner location of a check using the technique of hough lines.
"""

import cv2
import numpy as np
import math


def color_mask(frame): 
    """ This method applies a color mask on an image.  This is useful since 
    we may only care about boundaries in a particular color. """
    
    # Masks are easier to identify in HSV space.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # green color mask
    lower = np.uint8([20, 20, 20])
    upper = np.uint8([50, 255, 255])
    green_mask = cv2.inRange(frame, lower, upper)

    # white color mask
    lower = np.uint8([0, 0, 150])
    upper = np.uint8([255, 20, 255])
    white_mask = cv2.inRange(frame, lower, upper)

    # yellow color mask
    lower = np.uint8([13, 60, 150])
    upper = np.uint8([33, 255, 255])
    yellow_mask = cv2.inRange(frame, lower, upper)

    # red color mask 1
    lower = np.uint8([0, 60, 100])
    upper = np.uint8([15, 255, 255])
    red_mask_1 = cv2.inRange(frame, lower, upper)

    # red color mask 2
    lower = np.uint8([120, 100, 0])
    upper = np.uint8([255, 255, 255])
    red_mask_2 = cv2.inRange(frame, lower, upper)
    
    # combine the mask
    mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), red_mask_1), red_mask_2)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    green_masked = cv2.bitwise_and(frame, frame, mask=green_mask)

    masked = cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)
    green_masked = cv2.cvtColor(green_masked, cv2.COLOR_HSV2RGB)
    return masked, green_masked


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=50, high_threshold=250):
    return cv2.Canny(image, low_threshold, high_threshold)

def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

def group_lines(slopes, intercepts, lengths, slope_threshold, intercept_threshold):
    """ This method takes a set of lines and groups them based on their slopes and intercepts.  This prevents 
    a lot of extremely similar lines being sent along which will confuse the later processes. """

    # We group lines by looking at their slopes and intercepts.  If they are within x% of each other with both the slope
    # and intercept, then I will deem them to be the same line.
    
    line_group_counter = 1
    # line_groups stores group membership for each line.
    line_groups = []
    
    for line_index in range(0, len(slopes)):
        slope = slopes[line_index]
        intercept = intercepts[line_index]

        found_group_flag = 0
        
        if line_group_counter == 1:
            line_groups.append(1)
            line_group_counter += 1
            continue
        
        for line_group_index in range(1, line_group_counter):
            # Get the indices of the lines in the current group.
            indices = [i for i in range(len(line_groups)) if line_groups[i] == line_group_index]
            current_group_is_match_flag = 1


            # Use those indices to get the slopes and intercepts of all lines in the current group.
            slopes_current_group = [slopes[index] for index in indices]
            intercepts_current_group = [intercepts[index] for index in indices]

            # Insert for loop where we check the current line of interest with each line in the current group.
            # If similar to all of the lines, make the line_group of the current line of interest equal to the 
            # line group index.

            for element_in_group_index in range(0, len(slopes_current_group)):
                line_in_group_slope = slopes_current_group[element_in_group_index]
                line_in_group_intercept = intercepts_current_group[element_in_group_index]

                slope_percent_diff = (slope - line_in_group_slope) / (line_in_group_slope) * 100
                intercept_percent_diff = (intercept - line_in_group_intercept) / (line_in_group_intercept) * 100

                if (np.abs(slope_percent_diff) <= slope_threshold and np.abs(intercept_percent_diff) <= intercept_threshold):
                    continue
                else:
                    current_group_is_match_flag = 0
                    break

            # Now we analyze the value of found_group_flag.  If it is 1, then the current group is a good enough fit, so assign the group.
            if current_group_is_match_flag == 1:
                found_group_flag = 1
                line_groups.append(line_group_index)
                break
            else:
                # Keep searching for a group.
                continue

        if found_group_flag == 0:
            line_groups.append(line_group_counter)
            line_group_counter += 1

    # Now at this point, all lines have been assigned a line group.
    return slopes, intercepts, lengths, line_groups


def average_line_groups(slopes, intercepts, lengths, line_groups):
    """ Returns a list of tuples.  Each tuple is a line in form (slope, intercept) """
    averages_list = []

    # Get the number of groups we found.
    num_groups = max(line_groups)

    for group_number in range(1, num_groups+1):
        indices = [i for i in range(len(slopes)) if line_groups[i] == group_number]
        group_slopes = [slopes[index] for index in indices]
        group_intercepts = [intercepts[index] for index in indices]

        average_slope = np.average(group_slopes)
        average_intercept = np.average(group_intercepts)
        tup = (average_slope, average_intercept)

        averages_list.append(tup)
    
    return averages_list

def get_line_info(lines):
    """ Input is a  """
    slopes = []
    intercepts = []
    lengths = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)

            slopes.append(slope)
            intercepts.append(intercept)
            lengths.append(length)

    return slopes, intercepts, lengths

def find_lines(image, lines, slope_threshold_for_similarity, intercept_threshold_for_similarity):
    """ This method takes an image and a set of lines and returns a list of line endpoints after the lines have been
    averaged together. """
    slopes, intercepts, lengths = get_line_info(lines)
    slopes, intercepts, lengths, line_groups = group_lines(slopes, intercepts, lengths, slope_threshold=slope_threshold_for_similarity, intercept_threshold=intercept_threshold_for_similarity)

    if not line_groups:
        return [], -1

    averages = average_line_groups(slopes, intercepts, lengths, line_groups)

    # Get the y values of the endpoints of the lines we will display.
    y_bottom = image.shape[0]
    y_mid = y_bottom*0.6

    endpoints_list = []

    for line_index in range(0, len(averages)):
        line_tuple = averages[line_index]
        endpoints = find_line_endpoints(y_bottom, y_mid, line_tuple)
        endpoints_list.append(endpoints)

    return endpoints_list, len(averages)


def find_line_endpoints(y1, y2, line):
    """ Finds line points """
    if line is None:
        return None

    slope, intercept = line

    if (slope == 0.0 or math.isinf(slope) or math.isinf(intercept)):
        return None

    # make sure everything is integer as cv2.line requires it.
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))
    
def draw_lines(image, lines, color=[255, 0, 0], thickness=10):
    """ Takes a list of Hough Lines  """
    # make a separate image to draw lines and combine with the original later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
