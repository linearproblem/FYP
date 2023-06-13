import cv2
import numpy as np


def erode_and_dilate(img, erode_dimensions, dilate_dimensions):
    """Perform erosion and dilation of an image.
    This is 'opening' the image in morphology terms
    """
    mask = cv2.erode(img, np.ones(erode_dimensions, "uint8"))
    return cv2.dilate(mask, np.ones(dilate_dimensions, "uint8"))


def apply_clahe(img):
    # Check if image is grayscale, if not, convert
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check if image is of type uint8 or uint16, if not, convert
    if img.dtype != np.uint8 and img.dtype != np.uint16:
        img = img.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def canny_edge_detection(img, lower_thresh, upper_thresh):
    """Perform Canny edge detection and find contours."""
    edges = cv2.Canny(img, lower_thresh, upper_thresh, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(edges.shape)
    cv2.drawContours(img_contours, contours, -1, 255, 3)
    return img_contours.astype(np.uint8)
    #return edges


def get_edges(img, canny_lower_thresh, canny_upper_thresh):
    """Perform edge detection on an image."""
    img = cv2.medianBlur(erode_and_dilate(img, (8, 0), (32, 0)), 9)
    img = apply_clahe(img)
    return canny_edge_detection(img, canny_lower_thresh, canny_upper_thresh)


import cv2
import numpy as np


def distance_between_horizontal_lines(img_grey, region_top, region_bottom, return_frames=False):
    def convert_relative_to_pixel_coordinates(image, relative_coordinates):
        height, width = image.shape[:2]
        pixel_coordinates = []
        for coord in relative_coordinates:
            pixel_coordinates.append((int(coord[0] * width), int(coord[1] * height)))
        return tuple(pixel_coordinates)

    def find_all_lines_in_region(lines, angle, region):
        # unique_lines = filter_unique_lines(lines)
        unique_lines = lines
        return filter_lines_in_region(unique_lines, angle, region)

    def filter_unique_lines(lines):
        # Difference in pixels between lines for uniqueness
        thresh = 1
        # Gradient threshold, lines must be within this [lower,upper]
        grad_threshold = (-0.5, 0.5)
        unique_lines = []

        for i, ltest in enumerate([line[0] for line in lines]):
            # i.e. don't test on self or any already tested ones
            if is_duplicate_line(ltest, lines[:i], thresh, grad_threshold):
                continue
            unique_lines.append(ltest)
        return unique_lines

    def is_duplicate_line(ltest, lines, thresh, grad_threshold):
        return any(
            is_same_line(ltest, line, thresh) or not is_gradient_in_threshold(ltest, grad_threshold) for line in lines)

    def is_same_line(l1, l2, thresh):
        # The l1 and l2 arrays are lines in the format [x0, y0, x1, y1]
        # Where (x0, y0) and (x1, y1) are the coordinates of the start and end points of each line.

        # Ensure l1 and l2 are 1D arrays
        l1 = np.squeeze(l1)
        l2 = np.squeeze(l2)
        xm0, ym0 = np.abs(l1[0] - l2[0]), np.abs(l1[1] - l2[1])
        xm1, ym1 = np.abs(l1[2] - l2[2]), np.abs(l1[3] - l2[3])
        return (xm0 < thresh).all() and (ym0 < thresh).all() and (xm1 < thresh).all() and (ym1 < thresh).all()

    def is_gradient_in_threshold(l, grad_threshold):
        if (l[2] - l[0]) == 0:
            return False
        grad = (l[3] - l[1]) / (l[2] - l[0])
        return grad_threshold[0] <= grad <= grad_threshold[1]

    def filter_lines_in_region(lines, angle, region):
        lines_in_region = [line for line in lines if is_line_in_region(line, region, angle)]

        if not lines_in_region:
            print("No lines found in region")
            return None, None
        points_in_region = np.concatenate(np.concatenate(lines_in_region))
        points_x = points_in_region[0::2]
        points_y = points_in_region[1::2]
        return points_x, points_y

    def is_line_in_region(l, region, angle):
        l = np.squeeze(l)
        within_x_boundaries = False
        within_y_boundaries = False
        within_angle_boundaries = False
        if (l[0] > region[0][0]) & (l[0] < region[1][0]):  # x0 is within x boundaries
            if (l[2] < region[1][0]) & (l[2] > region[0][0]):  # x1 is within x boundaries
                within_x_boundaries = True
        if (l[1] > region[0][1]) & (l[1] < region[1][1]):  # y0 is within y boundaries
            if (l[3] < region[1][1]) & (l[3] > region[0][1]):  # y1 is within y boundaries
                within_y_boundaries = True

        if (l[1] - l[0]) != 0:
            theta = np.arctan((l[3] - l[2]) / (l[1] - l[0]))  # θ = arctan(Δy/Δx)
        else:
            theta = 90  # Above threshold
        theta = 180 * theta / np.pi
        within_angle_boundaries = -angle < theta < angle
        if within_x_boundaries:
            pass
        # return within_x_boundaries.all() and within_y_boundaries.all() and within_angle_boundaries
        return within_x_boundaries and within_y_boundaries and within_angle_boundaries

    def sort_lists(*lists):
        return [list(t) for t in zip(*sorted(zip(*lists)))]

    def remove_outliers(x, y):
        q75, q25 = np.percentile(y, [85, 15])
        buffer = 5
        nX, nY = zip(*[(xi, yi) for xi, yi in zip(x, y) if q25 - buffer < yi <= q75 + buffer])
        return list(nX), list(nY)

    # Edge detection
    edges = get_edges(img_grey,50,150)#cv2.Canny(img_grey, 50, 150, apertureSize=3)

    # Line detection
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=5, maxLineGap=50)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=5, maxLineGap=2)

    # convert relative region coordinates to pixel coordinates
    region_top = convert_relative_to_pixel_coordinates(img_grey, region_top)
    region_bottom = convert_relative_to_pixel_coordinates(img_grey, region_bottom)

    ## TODO: DEBUGGING FROM HERE

    if lines is None:
        return None, None

    angle = 45  # * np.pi / 180  # Convert angle to radians
    x_top, y_top = find_all_lines_in_region(lines, angle, region_top)
    x_bottom, y_bottom = find_all_lines_in_region(lines, angle, region_bottom)

    # Lines not found in regions
    return_flag = False
    if not isinstance(x_top, (np.ndarray, list, tuple)):
        if x_top is None:
            return_flag = True
    if not isinstance(x_bottom, (np.ndarray, list, tuple)):
        if x_bottom is None:
            return_flag = True

    if not isinstance(y_top, (np.ndarray, list, tuple)):
        if y_top is None:  # Lines not found in regions
            return_flag = True
    if not isinstance(y_bottom, (np.ndarray, list, tuple)):
        if y_bottom is None:
            return_flag = True

    if return_flag:
        return None, None
    try:
        x_top, y_top = sort_lists(x_top, y_top)
        x_bottom, y_bottom = sort_lists(x_bottom, y_bottom)
        x_top, y_top = remove_outliers(x_top, y_top)
        x_bottom, y_bottom = remove_outliers(x_bottom, y_bottom)
    except:
        print("cunt")

    if len(x_top) > 1:
        y_avg_top = sum(y_top) / len(y_top)
    else:
        return None, None

    if len(x_bottom) > 1:
        y_avg_bottom = sum(y_bottom) / len(y_bottom)
    else:
        return None, None

    distance_in_px = y_avg_bottom - y_avg_top
    if not return_frames:
        return distance_in_px, None

    # Create a blank image with the same dimensions as the original image
    lines_image = np.zeros_like(cv2.cvtColor(np.copy(edges), cv2.COLOR_GRAY2BGR))
    # Draw the lines on the lines_image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red color

    # Draw the regions
    cv2.rectangle(lines_image, tuple(region_top[0]), tuple(region_top[1]), (0, 255, 0),
                  2)  # Draw rectangle in green color
    cv2.rectangle(lines_image, tuple(region_bottom[0]), tuple(region_bottom[1]), (255, 0, 0),
                  2)  # Draw rectangle in blue color

    cv2.line(lines_image, (x_bottom[0], y_bottom[0]), (x_bottom[1], y_bottom[1]), (255, 0, 0), 10)
    cv2.line(lines_image, (x_top[0], y_top[0]), (x_top[1], y_top[1]), (255, 0, 0), 10)

    # Distance line
    cv2.line(lines_image, (x_top[0], y_top[0]), (x_top[0], int(y_top[0] + int(distance_in_px))), (0, 255, 255), 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return distance_in_px, np.hstack([edges, lines_image])


def is_cap_secure(bottle_frame, return_frame=False):
    cap_secure = False
    cap_region = ((0.29353932584269665, 0.07612293144208038), (0.8146067415730337, 0.12671394799054372))
    top_edge_region = ((0.11797752808988764, 0.19574468085106383), (0.8918539325842697, 0.2515366430260047))
    distance_in_px, frame = distance_between_horizontal_lines(bottle_frame, cap_region, top_edge_region, return_frame)

    if distance_in_px is not None:
        if distance_in_px > 240:
            cap_secure = True
    else:
        cap_secure = False

    if return_frame:
        return cap_secure, frame
    else:
        return None, None
