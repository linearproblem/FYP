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
            for q_rgb, device_id in q_rgb_map:
                frame = q_rgb.get().getCvFrame()

                if device_id == front_camera:
                    # run certain functions
                    pass
                elif device_id == rear_camera:
                    # run other functions
                    pass

            frames = [q_rgb.get().getCvFrame() for q_rgb, _ in q_rgb_map]

            # Define the callback function
            def on_click(row, col):
                print(f"Section ({row}, {col}) was clicked")


            # Call the function
            display_window(frames)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    print(read_error_log())
    clear_error_log()
