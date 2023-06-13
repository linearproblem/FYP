# Todo: look at the improvements recommended by ..
#           - Also try and add different status colours, such as active, inactive, failure, good. and possibly a legend

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Configure the logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
NUM_ROWS = 10
NUM_COLS = 6
SECTION_HEIGHT = int(WINDOW_HEIGHT / NUM_ROWS)
SECTION_WIDTH = int(WINDOW_WIDTH / NUM_COLS)
FONT_PATH = "C:/Windows/Fonts/Arial.ttf"
FONT_SIZE = 20
click_position = [None, None]  # Used to store section clicked in GUI


def pad_image(image, num_cols, num_rows, section_width, section_height):
    """
    Adjust the image size maintaining its aspect ratio.
    """
    aspect_ratio_sections = (num_cols * section_width) / (num_rows * section_height)
    img_height, img_width, _ = image.shape
    aspect_ratio_img = img_width / img_height

    # If the current aspect ratio is smaller than the target aspect ratio, add padding to the left and right
    if aspect_ratio_img < aspect_ratio_sections:
        new_width = int(img_height * aspect_ratio_sections)
        pad = (new_width - img_width) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, pad, new_width - img_width - pad, cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])

    # else the current aspect ratio is larger than the target aspect ratio, add padding to the top and bottom
    else:
        new_height = int(img_width / aspect_ratio_sections)
        pad = (new_height - img_height) // 2
        padded_image = cv2.copyMakeBorder(image, pad, new_height - img_height - pad, 0, 0, cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])
    return padded_image


def write_image_to_sections(image, start_row, start_col, num_rows, num_cols, section_width, section_height,
                            combined_sections, proportional_scale=False):
    """
    Write image into sections maintaining aspect ratio.
    """
    num_img_rows = start_row + num_rows
    num_img_cols = start_col + num_cols
    # Maintain Original Aspect ratio by padding image
    if proportional_scale:
        image = pad_image(image, num_cols, num_rows, section_width, section_height)
    # Ensure image fits within this many sections
    img = cv2.resize(image, (section_width * num_cols, section_height * num_rows))

    for i in range(start_row, num_img_rows):
        for j in range(start_col, num_img_cols):
            x = j * section_width
            y = i * section_height

            if y + section_height > combined_sections.shape[0] or x + section_width > combined_sections.shape[1]:
                continue

            combined_sections[y:y + section_height, x:x + section_width] = img[(i - start_row) * section_height:(
                                                                                                                        i - start_row + 1) * section_height,
                                                                           (j - start_col) * section_width:(
                                                                                                                   j - start_col + 1) * section_width]
    return combined_sections


def on_click(event, x, y, flags, params):
    """
    Callback for mouse click event.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position[0] = y // SECTION_HEIGHT
        click_position[1] = x // SECTION_WIDTH
        return None


def draw_text_with_alignment(draw, text, fill, align="left", valign="top", image_width=0, image_height=0,
                             padding=0):
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    text = text.replace('_', ' ').title()  # Reformat into Capital Case, replace underscores with spaces
    text_width, text_height = draw.textsize(text, font=font)

    if align == "left":
        text_x = padding
    elif align == "right":
        text_x = image_width - text_width - padding
    else:  # align == "centre"
        text_x = (image_width - text_width) // 2

    if valign == "top":
        text_y = padding
    elif valign == "bottom":
        text_y = image_height - text_height - padding
    else:  # valign == "middle" or valign == "mid"
        text_y = (image_height - text_height) // 2

    draw.text((text_x, text_y), text, font=font, fill=fill)


def create_section_images():
    section_images = []

    for i in range(NUM_ROWS):
        row_images = []
        for j in range(NUM_COLS):
            # Highlight active feature if there is one present
            if click_position != (None, None) and [i, j] == list(click_position) and not (i > 0 and j > 0):
                section = np.full((SECTION_HEIGHT, SECTION_WIDTH, 3), [170, 0, 60], dtype=np.uint8)
            else:
                # Just make the section a black box if feature not present
                section = np.zeros((SECTION_HEIGHT, SECTION_WIDTH, 3), dtype=np.uint8)
            row_images.append(section)
        section_images.append(row_images)
    return section_images


def add_section_text(section_images, feature_keys, active_status, front_camera_status, rear_camera_status):
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            # Convert the OpenCV image to a PIL Image
            pil_image = Image.fromarray(section_images[i][j])

            # Create a drawing context
            draw = ImageDraw.Draw(pil_image)

            # Write the section number/feature number to the image
            if i * NUM_COLS + j < len(feature_keys):
                quality_check_name = str(feature_keys[i * NUM_COLS + j])
                qc_colour = (255, 255, 255) if active_status[i * NUM_COLS + j] else (50, 50, 50)
                draw_text_with_alignment(draw, quality_check_name, qc_colour, valign='top', align='centre',
                                         image_width=SECTION_WIDTH, image_height=SECTION_HEIGHT, padding=10)
            else:
                section_text = f"Section ({i}, {j})"
                draw_text_with_alignment(draw, section_text, (50, 50, 50), valign='middle', align='centre',
                                         image_width=SECTION_WIDTH, image_height=SECTION_HEIGHT, padding=10)

            # Convert the PIL Image back to an OpenCV image and replace the original
            section_images[i][j] = np.array(pil_image)

            # Add a border to the section image
            cv2.rectangle(section_images[i][j], (0, 0), (SECTION_WIDTH - 1, SECTION_HEIGHT - 1), (255, 255, 255), 2)
    return section_images


def get_clicked_feature(feature_keys, active_status):
    """
    Used to handle when a feature is clicked on in the GUI and trigger a focus/looking at the feature directly
    :param feature_keys:
    :param active_status:
    :return:
    """
    try:
        #  Click position is None by default
        index = click_position[0] * NUM_COLS + click_position[1]
        if active_status[index]:
            return str(feature_keys[index])  # Returns the name of the feature to focus on
        else:
            return None
    except (TypeError, IndexError):
        return None


def display_window(camera_frames, quality_checks=None):
    """
    Display window with sections.
    """
    section_images = create_section_images()

    # Combine the front and rear camera features into one dictionary
    feature_keys = list(quality_checks.keys())
    active_status = [feature_data.get('active', False) for feature_data in quality_checks.values()]
    front_camera_status = [feature_data['location'].get('front', False) for feature_data in quality_checks.values()]
    rear_camera_status = [feature_data['location'].get('rear', False) for feature_data in quality_checks.values()]

    section_images = add_section_text(section_images, feature_keys, active_status, front_camera_status,
                                      rear_camera_status)
    combined_sections = np.vstack([np.hstack(row_images) for row_images in section_images])

    # Load the camera images into the window
    if camera_frames is not None:
        camera_frames = [frame_tuple[0] for frame_tuple in camera_frames]
        if len(camera_frames) > 0:
            if camera_frames[0] is not None:
                combined_sections = write_image_to_sections(camera_frames[0], 2, 0, 8, 3, SECTION_WIDTH, SECTION_HEIGHT,
                                                            combined_sections, proportional_scale=True)

        if len(camera_frames) > 1:
            if camera_frames[1] is not None:
                combined_sections = write_image_to_sections(camera_frames[1], 2, 3, 8, 3, SECTION_WIDTH, SECTION_HEIGHT,
                                                            combined_sections, proportional_scale=True)

    # Show the image
    cv2.namedWindow("My Window", cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("My Window", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setWindowProperty('My Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("My Window", combined_sections)

    # Here, we set up the event listener for mouse clicks, calling on_click
    cv2.setMouseCallback("My Window", on_click)

    return get_clicked_feature(feature_keys, active_status)
