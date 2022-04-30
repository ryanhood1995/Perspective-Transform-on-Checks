""" 
Name: find_initial_corners.py

Author: Ryan Hood
Email: ryanchristopherhood@gmail.com

Description: This file is part of a larger project which performs a perspective 
transformation on images of checks.  This file in particular provides methods useful 
for returning the location of the four corners of the check in the original image.
"""

import cv2
import hough_line_methods
import numpy as np


def find_contour_from_hough_lines(img):
    """ This method is called whenever the previous two methods fail. 
    TO DO: Fill this in using the methods in hough_line_methods.py. """
    # Apply Color Mask.
    
    # Blur Image.
    
    # Canny Edge Detector.
    
    # Hough Lines.
    
    # Group the Hough Lines.
    
    # Pick the best Hough Lines for describing Check.
    return
    


def find_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
        """ This method takes two points on two lines and finds the intersection of those two lines. """
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
        return [px, py]


def find_contour_from_convex_hull(img):
    """ This method takes an image and returns the coordinates of the corners of the check using convex hull method. """
    
    # Create a copy so we do not alter the original.
    image = img.copy()
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_scaled, (25,25), 0)
    edges = cv2.Canny(blurred, 25, 50, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, 60, maxLineGap=50)
    
    
    # Create image with just lines.
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    
    
    gray_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    # Find contours from just the hough lines.
    cnts = cv2.findContours(gray_line.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    c_img = np.zeros_like(image)

    cv2.drawContours(c_img, cnts, -1, 255, 1)
    

    # Find Hough Lines from contours
    hull_img = np.zeros_like(image)
    points3 = [pt[0] for ctr in cnts for pt in ctr]
    points3 = np.array(points3).reshape((-1,1,2)).astype(np.int32)
    hull3 = cv2.convexHull(points3)
    hull_img=cv2.drawContours(hull_img,[hull3],-1,(0,255,0),1,cv2.LINE_AA)  
    gray_hull = cv2.cvtColor(hull_img, cv2.COLOR_BGR2GRAY)

    #get lines from convex hull
    hull_lines = cv2.HoughLinesP(gray_hull, 1, np.pi/180, 200, 100, maxLineGap=10)
    

    top_line = hull_lines[0]
    bot_line = hull_lines[0]
    left_line = hull_lines[0]
    right_line = hull_lines[0]

    # Find the top line.
    if hull_lines is not None:
        for line in hull_lines:
            for x1, y1, x2, y2 in line:
                for tx1, ty1, tx2, ty2 in top_line:
                    avgy_line = int(y1+y2/2)
                    avgy_tline = int(ty1+ty2/2)
                    if (avgy_line < avgy_tline):
                        top_line = line

    # Find the bottom line.
    if hull_lines is not None:
        for line in hull_lines:
            for x1, y1, x2, y2 in line:
                for bx1, by1, bx2, by2 in bot_line:
                    avgy_line = int(y1+y2/2)
                    avgy_bline = int(by1+by2/2)
                    if (avgy_line > avgy_bline):
                        bot_line = line
                        
    # Find the left-most line.
    if hull_lines is not None:
        for line in hull_lines:
            for x1, y1, x2, y2 in line:
                for lx1, ly1, lx2, ly2 in left_line:
                    avgx_line = int(x1+x2/2)
                    avgx_lline = int(lx1+lx2/2)
                    if (avgx_line < avgx_lline):
                        left_line = line

    # Find the right-most line.
    if hull_lines is not None:
        for line in hull_lines:
            for x1, y1, x2, y2 in line:
                for rx1, ry1, rx2, ry2 in right_line:
                    avgx_line = int(x1+x2/2)
                    avgx_rline = int(rx1+rx2/2)
                    if (avgx_line > avgx_rline):
                        right_line = line
    
    
    # Now that we have the lines, we need to find the corners of the lines.
    for x1, y1, x2, y2 in top_line:
        for ax1, ay1, ax2, ay2 in left_line:
            tl_x, tl_y = find_intersection(x1, y1, x2, y2, ax1, ay1, ax2, ay2)

    for x1, y1, x2, y2 in top_line:
        for ax1, ay1, ax2, ay2 in right_line:
            tr_x, tr_y = find_intersection(x1, y1, x2, y2, ax1, ay1, ax2, ay2)

    for x1, y1, x2, y2 in bot_line:
        for ax1, ay1, ax2, ay2 in left_line:
            bl_x, bl_y = find_intersection(x1, y1, x2, y2, ax1, ay1, ax2, ay2)

    for x1, y1, x2, y2 in bot_line:
        for ax1, ay1, ax2, ay2 in right_line:
            br_x, br_y = find_intersection(x1, y1, x2, y2, ax1, ay1, ax2, ay2)

    tl = [tl_x, tl_y]
    bl = [bl_x, bl_y]
    tr = [tr_x, tr_y]
    br = [br_x, br_y]
    

    ordered_corner_points = np.array([tl, tr, br, bl], dtype="int32")
    ordered_corner_points = np.reshape(ordered_corner_points, (4,1,2))

    return ordered_corner_points, True



def find_contour_from_approxpolydp(img):
    """ This method takes an image and if possible returns a set of contours describing 
    the edge of the check.  The key restriction is that the inside area of the contour must be 
    some set amount and there must be exactly 4 contour lines. """
    found_contour = False
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Converting image to a binary image
    # (black and white only image).
    _, threshold = cv2.threshold(grayscale, 110, 255, cv2.THRESH_BINARY)
    
    # Detecting shapes in image by selecting region
    # with same colors or intensity.
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Searching through every region selected to
    # find the required polygon.
    for cnt_index in range(len(contours)):
        cnt = contours[cnt_index]
        area = cv2.contourArea(cnt)
    
        # Shortlisting the regions based on their area.
        if area > 100000:
            #0.03
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    
            # Checking if the no. of sides of the selected region is 7.
            if(len(approx) == 4):
                final_contour = approx
                found_contour = True
    
    try: 
        final_contour
    except NameError: 
        final_contour = None
    
    return final_contour, found_contour


def find_quadrilateral_contour(img):
    """ This method is the master method for finding the best contour representing the check. """
    
    final_contour, found_quad_using_approxpolydp = find_contour_from_approxpolydp(img)
    
    if found_quad_using_approxpolydp == False:
        final_contour, found_quad_using_convex_hull = find_contour_from_convex_hull(img)
        
        if found_quad_using_convex_hull == False:
            print("Methods failed on this check.")
            final_contour = find_contour_from_hough_lines(img)
            
    return final_contour
    
    
        

def extract_four_corners_from_contour(contour):
    """ This method takes a contour and extracts the corner points. """
    
    x1 = contour[1][0][0]
    y1 = contour[1][0][1]
    
    x2 = contour[0][0][0]
    y2 = contour[0][0][1]
    
    x3 = contour[2][0][0]
    y3 = contour[2][0][1]
    
    x4 = contour[3][0][0]
    y4 = contour[3][0][1]
    
    corners_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    return corners_list




