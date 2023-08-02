import cv2
import numpy as np
from object_detection_utils import crop_bounding_box


# A really simple example that shows the barcode being found and comparing edges to object detection edges.
def label_straightness_simple(frames):
    """
    Searches through each camera frame to find if a barcode is present.
    If barcode is found, return the cropped barcode frame and id of the camera it was found on.
    Then finds the biggest confidence detection of class 1 or 2 from the same camera.
    Draws bounding box for barcode and horizontal line between each edge of the barcode bounding box to the edge of the image.

    Parameters:
    frames (list): List of tuples, each containing a frame, device id, and detection list.
                   Each detection is an object with 'label', 'confidence', and 'bbox' attributes.

    Returns:
    Image: Returns the modified frame if barcode is found and class 1 or 2 detection is present.
           If not, returns the original frame.
    """
    # Check if frames is a list and non-empty
    if not isinstance(frames, list) or not frames:
        return None

    first_camera_frame = []
    first_camera_flag = True
    for frame, device_id, detections in frames:
        if first_camera_flag:
            first_camera_frame = frame
            first_camera_flag = False

        # For Barcodes, we are interested in class 0
        filtered_detections = [d for d in detections.detections if d.label == 0]

        # If no objects of interest were detected, return None
        if not filtered_detections:
            continue  # Move to next iteration of foor loop

        # Get the detection with the highest confidence
        barcode_detection = max(filtered_detections, key=lambda d: d.confidence)

        if True:

            max_confidence_frame = np.copy(frame)

            if max_confidence_frame is not None:
                # Calculate barcode bounding box coordinates in pixels
                barcode_x1 = int(barcode_detection.xmin * frame.shape[1])
                barcode_x2 = int(barcode_detection.xmax * frame.shape[1])
                barcode_y1 = int(barcode_detection.ymin * frame.shape[0])
                barcode_y2 = int(barcode_detection.ymax * frame.shape[0])

                # Draw the bounding box for the barcode on the cropped frame
                cv2.rectangle(max_confidence_frame, (barcode_x1, barcode_y1),
                              (barcode_x2, barcode_y2), (0, 255, 0), 5)
                # Draw a horizontal line between each edge of the barcode bounding box to the edge of the image
                cv2.line(max_confidence_frame, (barcode_x1, barcode_y1),
                         (0, barcode_y1), (254, 127, 45), 12)  # Left edge
                cv2.line(max_confidence_frame, (barcode_x2, barcode_y2),
                         (max_confidence_frame.shape[1], barcode_y2), (45, 127, 254), 12)  # Right edge
                return crop_bounding_box(np.copy(max_confidence_frame), detections)

    return first_camera_frame
