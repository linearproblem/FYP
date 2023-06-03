import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Configure the logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


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
    logging.error("within_write_image_to_sections_start")
    num_img_rows = start_row + num_rows
    num_img_cols = start_col + num_cols
    logging.error("proportional_scale_start")
    # Maintain Original Aspect ratio by padding image
    if proportional_scale:
        image = pad_image(image, num_cols, num_rows, section_width, section_height)
    logging.error("proportional_scale_end----Resize_start")
    # Ensure image fits within this many sections
    img = cv2.resize(image, (section_width * num_cols, section_height * num_rows))
    logging.error("resize_end")
    logging.error("Start_for_loop")
    for i in range(start_row, num_img_rows):
        for j in range(start_col, num_img_cols):
            x = j * section_width
            y = i * section_height

            if y + section_height > combined_sections.shape[0] or x + section_width > combined_sections.shape[1]:
                continue
            logging.error("combined_sections_start")
            combined_sections[y:y + section_height, x:x + section_width] = img[(i - start_row) * section_height:(i - start_row + 1) * section_height,
                                                                              (j - start_col) * section_width:(j - start_col + 1) * section_width]
            logging.error("combined_sections_end----end_for_loop")
    logging.error("end_for_loop")
    logging.error("within_write_image_to_sections_end")
    return combined_sections


def on_click(event, x, y, section_height, section_width):
    """
    Callback for mouse click event.
    """
    logging.error("within_click_start")
    if event == cv2.EVENT_LBUTTONDOWN:
        row = y // section_height
        col = x // section_width
        return [row, col]
    logging.error("within_click_end")

def display_window(camera_frames):
    """
    Display window with sections.
    """
    logging.error("within_displayWindow_start")
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 800
    NUM_ROWS = 10
    NUM_COLS = 6
    SECTION_HEIGHT = int(WINDOW_HEIGHT / NUM_ROWS)
    SECTION_WIDTH = int(WINDOW_WIDTH / NUM_COLS)
    section_images = []
    combined_sections = []

    for i in range(NUM_ROWS):
        row_images = []
        for j in range(NUM_COLS):
            section = np.zeros((SECTION_HEIGHT, SECTION_WIDTH, 3), dtype=np.uint8)
            row_images.append(section)
        section_images.append(row_images)

    cv2.namedWindow("My Window", cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("My Window", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setWindowProperty('My Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Define the font for the text
    font = ImageFont.load_default()

    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            # Convert the OpenCV image to a PIL Image
            pil_image = Image.fromarray(section_images[i][j])

            # Create a drawing context
            draw = ImageDraw.Draw(pil_image)

            # Write the section number to the image
            section_text = f"Section ({i}, {j})"
            draw.text((10, SECTION_HEIGHT // 2), section_text, font=font, fill=(255, 255, 255))

            # Convert the PIL Image back to an OpenCV image and replace the original
            section_images[i][j] = np.array(pil_image)

            # Add a border to the section image
            cv2.rectangle(section_images[i][j], (0, 0), (SECTION_WIDTH - 1, SECTION_HEIGHT - 1), (255, 255, 255), 2)

    # Combine the section images into a single image - making up the whole menu
    combined_sections = np.vstack([np.hstack(row_images) for row_images in section_images])
    logging.error("before_write_image_to_new_sections")
    # Load the camera images into the window
    if len(camera_frames) > 0 and camera_frames[0] is not None:
        combined_sections = write_image_to_sections(camera_frames[0], 2, 0, 8, 3, SECTION_WIDTH, SECTION_HEIGHT, combined_sections, proportional_scale=True)

    if len(camera_frames) > 1 and camera_frames[1] is not None:
        combined_sections = write_image_to_sections(camera_frames[1], 2, 3, 8, 3, SECTION_WIDTH, SECTION_HEIGHT, combined_sections, proportional_scale=True)

    # Show the image
    cv2.imshow("My Window", combined_sections)
    logging.error("within_displayWindow_end")
    return None
