# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # Todo:
    #   - Setup imports
    #   - Define locations for files
    #   - (menu will eventually go here)
    #   - Get Settings (hardcode bottle for now)
    #   - Setup Camera
    #   - Setup camera links

    # Standard library imports
    import os
    import sys

    # Related third-party imports
    import numpy as np
    import contextlib
    import cv2

    # Local imports
    import camera_setup
    from log_utils import read_error_log, clear_error_log
    from gui_window import display_window

    # Variables
    # Camera IDs
    front_camera = "19443010B118F81200"  # Front (Oak-D Lite)
    rear_camera = "19443010F19B281300"  # Rear (Oak-1)

    # Call the function and store the returned values
    quality_checks = config_parser.get_bottle_settings('settings/bottle_settings.yaml')

    # Setup
    with contextlib.ExitStack() as stack:
        devices = []
        q_rgb_map = []

        for device_id in [front_camera, rear_camera]:
            device, q_rgb = camera_setup.setup_camera(device_id)
            if device is None or q_rgb is None:  # Camera not found
                continue
            device = stack.enter_context(device)  # ensure device is kept alive and cleaned up properly
            devices.append(device)
            q_rgb_map.append((q_rgb, device_id))  # Include the device_id for later reference
        while True:
            # From each camera, get an image frame, the device ID and the detected objects (if objects detected)
            frames = [(q_rgb.get().getCvFrame(), device_id, q_detection.get() if q_detection.has() else None)
                      for q_rgb, device_id, q_detection in q_rgb_map]

            # Todo: see example code to figure out if a buffer might be useful, may need to do some testing
            for frame, device_id, detection in frames:
                bottle_frame = crop_bounding_box(np.copy(frame), detection)
                barcode_frame = crop_bounding_box(np.copy(frame), detection, barcode=True)

                if device_id == front_camera:
                    # run certain functions
                    frame = draw_bounding_box(frame, detection)
                    pass
                elif device_id == rear_camera:
                    # run other functions
                    frame = draw_bounding_box(frame, detection)
                    pass

                current_time = time.time()
                if current_time - last_time >= 1:  # A second has passed
                    print("Less than a second has passed since last run")
                    if active_feature == 'barcode':
                        _, active_frame = decode_barcode(barcode_frame, return_frame=True)
                    elif active_feature == 'bottle_orientation':
                        _, active_frame = is_bottle_upright(bottle_frame, return_frame=True)
                        pass
                else:  # Less than a second has passed
                    pass

                if active_feature is not None:
                    frames[0] = (active_frame, frames[0][1], frames[0][2])

            # Display the images in the window and get clicked feature if applicable
            active_feature = display_window(frames, quality_checks)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
            # Save frame when 's' key is pressed
            elif key == ord('s'):
                cv2.imwrite('frame.tiff', bottle_frame)  # Save the bottle frame from last camera

    print(read_error_log())
    clear_error_log()
