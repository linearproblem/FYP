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
