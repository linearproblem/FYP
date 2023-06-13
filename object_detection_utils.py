import cv2
import numpy as np
from settings.camera_id import front_camera, rear_camera


def draw_bounding_box(frame, detections, barcode=False):
    # Make sure there are detections
    if detections is None:
        return frame

    # Check if we are detecting Barcodes or Bottles
    if barcode:
        # For Barcodes, we are interested in class 0
        filtered_detections = [d for d in detections.detections if d.label == 0]
        bbox_color = (255, 0, 0)  # Blue color for barcode
    else:
        # For Bottles, we are interested in classes 1 and 2
        filtered_detections = [d for d in detections.detections if d.label in {1, 2}]
        bbox_color = (0, 0, 255)  # Red color for bottles

    # If no objects of interest were detected, return the original frame
    if not filtered_detections:
        return frame

    # Get the detection with the highest confidence
    detection = max(filtered_detections, key=lambda d: d.confidence)

    # Calculate bounding box coordinates in pixels
    x1 = int(detection.xmin * frame.shape[1])
    x2 = int(detection.xmax * frame.shape[1])
    y1 = int(detection.ymin * frame.shape[0])
    y2 = int(detection.ymax * frame.shape[0])

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 8)
    return frame


def draw_bounding_boxes_on_frames(captured_frames):
    """
    Draw bounding boxes on each frame in-place.

    Parameters:
    captured_frames (list): List of tuples, each containing a frame, device id, and detection.
    """
    if not isinstance(captured_frames, list):
        return

    for i, (frame, device_id, detection) in enumerate(captured_frames):
        frame = draw_bounding_box(frame, detection)
        captured_frames[i] = (frame, device_id, detection)


def crop_bounding_box(frame, detections, barcode=False):
    # Make sure there are detections
    if detections is None:
        return None

    # Check if we are detecting Barcodes or Bottles
    if barcode:
        # For Barcodes, we are interested in class 0
        filtered_detections = [d for d in detections.detections if d.label == 0]
    else:
        # For Bottles, we are interested in classes 1 and 2
        filtered_detections = [d for d in detections.detections if d.label in {1, 2}]

    # If no objects of interest were detected, return None
    if not filtered_detections:
        return None

    # Get the detection with the highest confidence
    detection = max(filtered_detections, key=lambda d: d.confidence)

    # Calculate bounding box coordinates in pixels
    x1 = int(detection.xmin * frame.shape[1])
    x2 = int(detection.xmax * frame.shape[1])
    y1 = int(detection.ymin * frame.shape[0])
    y2 = int(detection.ymax * frame.shape[0])

    # Crop the frame using the bounding box coordinates
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame


def find_barcode(frames):
    """
    Searches through each camera frame to find if a barcode is present.
    If barcode is found, return the cropped barcode frame and id of the camera it was found on.

    Parameters:
    frames (list): List of tuples, each containing a frame, device id, and detection.

    Returns:
    tuple: Cropped frame with the barcode and the device id of the camera it was found on.
           If no barcode is found, returns (None, None).
    """
    # Check if frames is a list and non-empty
    if not isinstance(frames, list) or not frames:
        return None, None

    for frame, device_id, detection in frames:
        barcode_frame = crop_bounding_box(np.copy(frame), detection, barcode=True)
        if barcode_frame is not None:
            return barcode_frame, device_id

    return None, None


def process_frames(captured_frames):
    """
    Processes each captured frame and returns a tuple of frames for each camera.

    Parameters:
    captured_frames (list): List of tuples, each containing a frame, device id, and detection.

    Returns:
    tuple: A tuple containing the front bottle frame, rear bottle frame.
    """
    if not isinstance(captured_frames, list) or not captured_frames:
        return None, None

    front_bottle_frame = None
    rear_bottle_frame = None

    for frame, device_id, detection in captured_frames:
        cropped_frame = crop_bounding_box(np.copy(frame), detection)
        if device_id == front_camera:
            front_bottle_frame = cropped_frame
        elif device_id == rear_camera:
            rear_bottle_frame = cropped_frame

    return front_bottle_frame, rear_bottle_frame



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
            continue # Move to next iteration of foor loop

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
