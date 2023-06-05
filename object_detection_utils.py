import cv2


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


def crop_bounding_box(frame, detections, barcode=False):
    # Make sure there are detections
    if detections is None:
        return frame

    # Check if we are detecting Barcodes or Bottles
    if barcode:
        # For Barcodes, we are interested in class 0
        filtered_detections = [d for d in detections.detections if d.label == 0]
    else:
        # For Bottles, we are interested in classes 1 and 2
        filtered_detections = [d for d in detections.detections if d.label in {1, 2}]

    # If no objects of interest were detected, return None
    if not filtered_detections:
        return frame

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
