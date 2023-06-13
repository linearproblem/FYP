import cv2
import numpy as np

# Constants
SMOOTHING_WINDOW_SIZE = 20  # Adjust this to change smoothing level
GRADIENT_WINDOW_SIZE = 5


def apply_contrast_stretching(img, region=None, out_min=0, out_max=255):
    import numpy as np

    if region is None:  # When region not specified do entire image
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]
    else:
        x, y, w, h = region  # extract region data

    roi = img[y:y + h, x:x + w]

    in_min = np.min(roi)
    in_max = np.max(roi)

    stretched_roi = ((roi - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min
    stretched_roi = stretched_roi.astype(np.uint8)

    return stretched_roi


def moving_median(arr, window_size):
    # Smooth out the median
    extended_arr = np.pad(arr, (window_size // 2, window_size - window_size // 2), mode='edge')

    # Find the median across each window
    result = np.array([np.median(extended_arr[i:i + window_size]) for i in range(len(arr))])
    return result


def moving_mean(arr, window_size):
    extended_arr = np.pad(arr, (window_size // 2, window_size - window_size // 2), mode='edge')
    # Find the mean across each window
    result = np.array([np.mean(extended_arr[i:i + window_size]) for i in range(len(arr))])
    return result


def find_liquid_location(equalised_image):
    """
    Returns the y-coordinate of the liquid's location in the image by finding
    the minimum of the smoothed gradient of pixel intensities.
    """
    # Calculate pixel intensity and apply smoothing
    y_intensity = np.sum(equalised_image, axis=1)

    # Smooth the y-axis pixel intensity using a moving median filter
    smoothed_intensity = moving_median(y_intensity, SMOOTHING_WINDOW_SIZE)

    # Compute mean of smoothed intensity and apply smoothing
    smoothed_median = moving_mean(np.gradient(smoothed_intensity), GRADIENT_WINDOW_SIZE)

    # Find and return the y-coordinate of the liquid - the global minimum gradient
    return np.argmin(smoothed_median)


def evaluate_bottle_fill(bottle_frame, return_frame=True):
    # Convert the input image to a grayscale image
    image_grey = cv2.cvtColor(bottle_frame, cv2.COLOR_BGR2GRAY)

    # Specify the region of interest to find the fill level
    x0 = 0.08357771260997067
    x1 = 0.19794721407624633
    y0 = 0.23061119671289163
    y1 = 0.6666666666666666

    frame_height, frame_width, _ = bottle_frame.shape
    x_start = int(x0 * frame_width)
    y_start = int(y0 * frame_height)
    x_end = int(x1 * frame_width)
    y_height = int((y1 - y0) * frame_height)

    x_width = 8  # Set the region width in pixels
    indexes = []
    # Iterate over the image along the x region
    for x in range(x_start, x_end - x_width + 1, x_width):
        for y in range(y_start, y_start + y_height, y_height):
            # Extract the sub-region
            sub_image = image_grey[y:y + y_height, x:x + x_width]
            img_contrast = apply_contrast_stretching(sub_image)
            indexes.append(find_liquid_location(img_contrast))

    # Find the median of each image section
    max_gradient_index = int(np.median(indexes) + y_start)

    # Using normalised height and expected fill regions
    is_filled_properly = 0.6 < (max_gradient_index / frame_width) < 0.7

    fill_frame = None
    if return_frame:
        fill_frame = np.copy(bottle_frame)
        # Draw a red line across the image at the Y location with the highest gradient
        cv2.line(fill_frame, (0, max_gradient_index), (fill_frame.shape[1] - 1, max_gradient_index), (0, 0, 255), 20)

        # Draw the rectangle using the specified values
        y_end = y_start + y_height
        cv2.rectangle(fill_frame, (x_start, y_start), (x_end, y_end), (255, 0, 255), 5)

    # bottle_frame = cv2.resize(bottle_frame, None, fx=0.2, fy=0.2)
    # cv2.imshow("Fill Level", bottle_frame)
    # key = cv2.waitKey(0)
    # if key in [ord('q'), ord('Q'), 27]:
    #     cv2.destroyAllWindows()

    return is_filled_properly, fill_frame
