import argparse
import os
import cv2
import numpy as np
import math

# ========================================================================
# Methods from Project 1
# ========================================================================

def color_mask(frame): 
    """ """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # I present, the perfect color masks for yellow, white, and red lane lines in HSV color space.

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
    """ Convert an image to grayscale. """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image, kernel_size=3):
    """ Perform Gaussian Blurring on an image. """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=50, high_threshold=250):
    """ Performs Canny Edge Detection on an image. """
    return cv2.Canny(image, low_threshold, high_threshold)


def remove_bad_lines(slopes, intercepts, lengths, slope_threshold, intercept_threshold):
    """ Lane lines have steeper slopes.  So if the absolute value of the slope of a line average is less than the slope
    threshold, we can remove that line.  We may do something with the intercept threshold later, but not for now."""
    indices_to_remove = []

    for index in range(0, len(slopes)):
        slope = slopes[index]
        if np.abs(slope) < slope_threshold:
            indices_to_remove.append(index)

    # Now remove based on indices.
    new_slopes = []
    new_intercepts = []
    new_lengths = []

    for original_index in range(0, len(slopes)):
        if original_index not in indices_to_remove:
            new_slopes.append(slopes[original_index])
            new_intercepts.append(intercepts[original_index])
            new_lengths.append(lengths[original_index])

    return new_slopes, new_intercepts, new_lengths

def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    """ Takes the output of a canny edge detector and finds the most likely straight lines. """
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def group_lines(slopes, intercepts, lengths, slope_threshold, intercept_threshold):
    """ This method takes a set of lines and groups them based on their slopes and intercepts. """


    # Now, We group lines by looking at their slopes and intercepts.  If they are within x% of each other with both the slope
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
    """ Returns a list of tuples.  Each tuple is a lane line in form (slope, intercept) """
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

def find_lane_lines(image, lines, slope_threshold_for_similarity, intercept_threshold_for_similarity):
    """ This method takes an image and a set of lines and returns a list of line endpoints after the lines have been
    averaged together. """
    slopes, intercepts, lengths = get_line_info(lines)
    slopes, intercepts, lengths = remove_bad_lines(slopes, intercepts, lengths, slope_threshold=1, intercept_threshold=5)
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
    
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    """ Takes a list of Hough Lines  """
    # make a separate image to draw lines and combine with the original later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)



# ========================================================================
# Methods
# ========================================================================


def find_contour_from_hough_lines(img):
    """ This method is called whenever the previous two methods fail. """
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
    """  """
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
    


# ========================================================================
# Drivers
# ========================================================================

def process_img(img_path):
    """ This method processes a single image, and shows both the original image and the new image. """
    img_original = cv2.imread(img_path)
    
    print("Processing image located here: ", img_path)
    
    # STEP 1: Find the corners of the original check.  Also find the target corners.
    original_contour = find_quadrilateral_contour(img_original)
    unordered_corners = extract_four_corners_from_contour(original_contour)
    
    is_landscape = determine_landscape(unordered_corners)
    
    source_corners = configure_source_corners(unordered_corners, is_landscape)
    target_corners = configure_resulting_corners(is_landscape)
    
    # STEP 2: Perform the prospective transform.
    corrected_image = perform_perspective_transform(img_original, source_corners, target_corners)
        
    # STEP 3: Clean up the resulting corrected image.    
    if is_landscape == 0:    
        cropped = corrected_image[20:1500,20:700]
        landscape_check = rotate_image(cropped)
    else:
        cropped = corrected_image[20:700,20:1500]
        landscape_check = cropped
        
    upright_check = make_check_upright(landscape_check)
    
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
            