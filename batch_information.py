import cv2
import re
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gui_window import FONT_PATH

def process_dot_matrix_print(bgr_image):
    # Convert the image to gray scale
    grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding on the image

    binary = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 8)
    binary = cv2.fastNlMeansDenoising(binary, h=10, templateWindowSize=7, searchWindowSize=21)

    # Defining a kernel for erosion and dilation
    kernel = np.ones((2, 2), np.uint8)

    # Using erosion followed by dilation to connect the dots in dot matrix prints
    img_erosion = cv2.erode(binary, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    # Convert the processed binary image back to BGR
    bgr_result = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR)

    return bgr_result


def detect_batch_expiry_and_best_before(frame, return_frame=False):
    if frame is None:
        return None, None

    # Normalised co-ordinate values, suitable for 500mL Square bottles (only ones given with batch numbers, expiry and best before dates)
    x0 = 0.08
    x1 = 0.96
    y0 = 0.20  # Made a bit taller to have room to fit the text
    #y1 = 0.30
    y1 = 0.40 # Changed when at Whiteley (after demo day) as camera angle is steeper

    # x0, x1, y0, y1 = (0, 1, 0, 1)

    # Convert normalised coordinates to pixel values
    y0_px = int(y0 * frame.shape[0])
    y1_px = int(y1 * frame.shape[0])
    x0_px = int(x0 * frame.shape[1])
    x1_px = int(x1 * frame.shape[1])

    # Crop the image
    info_frame = frame[y0_px:y1_px, x0_px:x1_px]

    # If frame is nothing after cropping
    if info_frame is None:
        return None, None

    preprocessed_info_frame = process_dot_matrix_print(np.copy(info_frame))
    info_frame = cv2.resize(info_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_info_frame = cv2.resize(preprocessed_info_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR
    for frame in [preprocessed_info_frame, info_frame]:
        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ:1234567890/ --psm 6 --oem 3 '
        extracted_text = pytesseract.image_to_string(frame, config=custom_config)
        # Check with regular expressions
        batch_number_match = re.search(r'BN([A-Z0-9]+)', extracted_text)
        expiry_date_match = re.search(r'EXP([0-9/]+)', extracted_text)
        best_before_match = re.search(r'BB([0-9/]+)', extracted_text)

        if batch_number_match is not None and expiry_date_match is not None and best_before_match is not None:
            break

    if batch_number_match is not None and expiry_date_match is not None and best_before_match is not None:
        batch_number = batch_number_match.group(1)
        expiry_date = expiry_date_match.group(1)
        best_before_date = best_before_match.group(1)
        info_found = True
    else:
        info_found = False



    if return_frame:
        if info_found:
            # Convert to PIL Image
            info_frame_pil = Image.fromarray(cv2.cvtColor(preprocessed_info_frame, cv2.COLOR_BGR2RGB))

            # Create a draw object
            draw = ImageDraw.Draw(info_frame_pil)
            font_size = int(info_frame_pil.height * 0.12)  # Scale font size to be 10% of image height
            font = ImageFont.truetype(FONT_PATH, font_size)

            # The starting point for the text, position text in top 10% of the image
            x = int(info_frame_pil.width * 0.01)  # Start at left corner
            y = int(info_frame_pil.height * 0.01)

            # Write batch number, expiry date and best before date to image
            for info_text in [f"BATCH: {batch_number}", f"EXPIRY: {expiry_date}", f"BEST BEFORE: {best_before_date}"]:
                text_width, text_height = draw.textsize(info_text, font=font)
                outline_color = "white"
                for adj in [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]:
                    draw.text(adj, info_text, font=font, fill=outline_color)
                draw.text((x, y), info_text, font=font, fill="magenta")
                y += text_height  # move to new line for next info

            # Convert back to OpenCV format
            preprocessed_info_frame = cv2.cvtColor(np.array(info_frame_pil), cv2.COLOR_RGB2BGR)

        frame = np.vstack([preprocessed_info_frame, info_frame])

    if return_frame:
        return info_found, frame
    else:
        return info_found, None
    # return info_found, None if not return_frame else frame


