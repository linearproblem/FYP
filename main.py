# Standard library imports
import os
import sys
import time

# Related third-party imports
import numpy as np
import contextlib
import cv2

# Local imports
from settings.camera_id import front_camera, rear_camera
import camera_setup
import config_parser
from log_utils import read_error_log, clear_error_log
from gui_window import display_window
from object_detection_utils import process_frames, find_barcode, draw_bounding_boxes_on_frames
from label_straightness import label_straightness_simple

# Quality Checks
from barcode_decoder import decode_barcode
from bottle_orientation import is_bottle_upright
from fill_level import evaluate_bottle_fill
from artg_id import detect_artg_id
from bottle_cap import distance_between_horizontal_lines, is_cap_secure
from batch_information import detect_batch_expiry_and_best_before

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Use for testing individual functions with a saved image
    if False:  # Allows enabling/disabling easily
        image = cv2.imread('./saved_images/front_frame_green.tiff')
        _, resulting_frame = is_bottle_upright(image, return_frame=True)
        print(result)
        while True:
            cv2.imshow("window", resulting_frame)
            cv2.waitKey(1)
        exit()

    # Call the function and store the returned values
    quality_checks = config_parser.get_bottle_settings('settings/bottle_settings.yaml')

    # Setup
    devices = []
    q_rgb_map = []
    front_bottle_frame = None
    rear_bottle_frame = None
    active_feature = None
    active_frame = None  # Store frame of current active feature
    barcode_camera_id = None  # Stores the camera that has found the barcode

    with contextlib.ExitStack() as stack:
        for device_id in [front_camera, rear_camera]:
            device, q_rgb, q_detection = camera_setup.setup_camera(device_id)  # Device and data queues
            if device is None or q_rgb is None or q_detection is None:  # Camera not found/queue error
                continue
            device = stack.enter_context(device)  # ensure device is kept alive and cleaned up properly
            devices.append(device)
            q_rgb_map.append((q_rgb, device_id, q_detection))

        while True:
            # From each camera, get an image frame, the device ID and the detected objects (if objects detected)
            frames = [(q_rgb.get().getCvFrame(), device_id, q_detection.get() if q_detection.has() else None)
                      for q_rgb, device_id, q_detection in q_rgb_map]

            front_bottle_frame, rear_bottle_frame = process_frames(frames)
            barcode_frame, barcode_camera_id = find_barcode(frames)
            draw_bounding_boxes_on_frames(frames)

            if front_bottle_frame is not None and rear_bottle_frame is not None:
                if active_feature == 'barcode' and barcode_frame is not None:
                    _, active_frame = decode_barcode(barcode_frame, return_frame=True)
                elif active_feature == 'bottle_orientation':
                    _, active_frame = is_bottle_upright(rear_bottle_frame, return_frame=True)
                    pass
                elif active_feature == 'fill_level':
                    _, active_frame = evaluate_bottle_fill(rear_bottle_frame, return_frame=True)
                    pass
                elif active_feature == 'artg_id' and barcode_camera_id is not None:
                    # Choose camera that is on the opposite side to the barcode
                    selected_frame = rear_bottle_frame if barcode_camera_id == front_camera else front_bottle_frame
                    _, active_frame = detect_artg_id(selected_frame, return_frame=True)
                    pass
                elif active_feature == 'cap_secure':
                    _, active_frame = is_cap_secure(front_bottle_frame, return_frame=True)
                elif active_feature == 'best_before':
                    selected_frame = rear_bottle_frame if barcode_camera_id == rear_camera else front_bottle_frame
                    _, active_frame = detect_batch_expiry_and_best_before(selected_frame, return_frame=True)
                elif active_feature == 'label_straightness':
                    selected_frame = rear_bottle_frame if barcode_camera_id == rear_camera else front_bottle_frame
                    active_frame = label_straightness_simple(frames)
                    None
                else:
                    active_frame = None

            if active_feature is not None and active_frame is not None:
                frames[0] = (active_frame, frames[0][1], frames[0][2])

            # Display the images in the window and get clicked feature if applicable
            active_feature = display_window(frames, quality_checks)

            # Close window when q,Q or Esc key pressed
            key = cv2.waitKey(1)
            if key in [ord('q'), ord('Q'), 27]:
                cv2.destroyAllWindows()
                break
            # Save frame when 'f' (front), 'r' (rear) or 'a' (active frame)  key is pressed
            elif key == ord('f'):
                cv2.imwrite('saved_images/front_frame.tiff', front_bottle_frame)
            elif key == ord('r'):
                cv2.imwrite('saved_images/rear_frame.tiff', rear_bottle_frame)
            elif key == ord('a'):
                current_time = time.time()  # Initialise to current time
                cv2.imwrite('C:/Users/micro/Desktop/saved_images/' + str(current_time) + str(
                    active_feature) + '_active_frame.tiff', selected_frame)
    print(read_error_log())
    clear_error_log()
