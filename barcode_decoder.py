import cv2
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
from pyzbar.pyzbar import decode, ZBarSymbol
import pytesseract
from gui_window import FONT_PATH

import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\OCR\Tesseract-OCR\tesseract.exe'  # Required for Windows

# Barcode-product mapping
barcode_to_product = {
    "9335006004129": "Dermalux Everyday 1L",
    "9335006006093": "Sanitol Jade 500mL",
    "9335006000060": "Bactol Blue 500mL",
    "9335006005348": "Bactol Clear 500mL",
    "9335006003115": "Sanitol Macadamia 500mL",
    "9000000000001": "Product Name",
    # add more barcodes and product names as needed
}


def desaturate_colorful_frame(image, brightness_factor=50):
    # Convert the image to HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Split the HSL image into channels
    h_channel, l_channel, s_channel = cv2.split(hsl_image)

    # Calculate the average saturation value
    mean_saturation = np.mean(s_channel)

    # Automatically determine the threshold for saturation
    threshold_saturation = int(mean_saturation * 0.2)

    # Threshold the saturation channel to identify colorful areas
    _, saturation_mask = cv2.threshold(s_channel, threshold_saturation, 255, cv2.THRESH_BINARY)

    # Apply brightness adjustment to the lightness channel for the identified areas
    l_channel_adjusted = cv2.add(l_channel, brightness_factor)

    # Merge the adjusted lightness channel with the original hue and saturation channels
    hsl_image_adjusted = cv2.merge((h_channel, l_channel_adjusted, s_channel))

    # Convert the adjusted HSL image back to the BGR color space
    bgr_image_adjusted = cv2.cvtColor(hsl_image_adjusted, cv2.COLOR_HLS2BGR)

    return bgr_image_adjusted


def apply_clahe(image, clip_limit=4.0, tile_grid_size=(2, 16)):
    """
    Contrast Limited Adaptive Histogram Equalisation
    Provides histogram equalisation over a small region,
    vertically focused due to barcode orientation.

    Regions are interpolated to reduce noice.
    """
    # Convert the BGR image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE on the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the A and B channels
    lab_image_clahe = cv2.merge((clahe_image, a_channel, b_channel))

    # Convert the LAB image with CLAHE back to BGR color space
    bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

    return bgr_image_clahe


def preprocess_barcode(frame, brightness=50, contrast=1.5):

    # Convert ROI to LAB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split LAB image into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(frame)

    # Apply brightness and contrast adjustments to the L channel
    l_channel = cv2.convertScaleAbs(l_channel, alpha=contrast, beta=brightness)

    # Merge the modified channels
    roi_lab_adjusted = cv2.merge((l_channel, a_channel, b_channel))

    # Convert the adjusted ROI back to BGR color space
    frame = cv2.cvtColor(roi_lab_adjusted, cv2.COLOR_LAB2BGR)

    # Desaturate barcode
    frame = desaturate_colorful_frame(frame)

    # Apply contrast limited histogram equalisation
    frame = apply_clahe(frame)
    return frame


def annotate_barcode_image(barcode_frame, decoded_barcode):
    # Convert the OpenCV image format to PIL image format
    barcode_frame_pil = Image.fromarray(cv2.cvtColor(barcode_frame, cv2.COLOR_BGR2RGB))
    # Create a draw object
    draw = ImageDraw.Draw(barcode_frame_pil)
    font_size = int(barcode_frame_pil.height * 0.1)  # Scale font size to be 10% of image height
    font = ImageFont.truetype(FONT_PATH, font_size)

    # The starting point for the text, position text in top 10% of the image
    x = int(barcode_frame_pil.width * 0.01)  # Start at left corner
    y = int(barcode_frame_pil.height * 0.05)

    # Write barcode number to image
    text = decoded_barcode
    text_width, text_height = draw.textsize(text, font=font)
    outline_color = "white"
    for adj in [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]:
        draw.text(adj, text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill="magenta")  # change "magenta" to any desired color

    # Lookup product name and use it if found
    product_name = barcode_to_product.get(decoded_barcode, "Product not found")
    product_text = f"{product_name}"

    # Write product name below barcode, on a new line
    y += text_height
    for adj in [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]:
        draw.text(adj, product_text, font=font, fill=outline_color)
    draw.text((x, y), product_text, font=font, fill="purple")  # change "purple" to any desired color

    # Convert the PIL image format back to OpenCV image format
    barcode_frame = cv2.cvtColor(np.array(barcode_frame_pil), cv2.COLOR_RGB2BGR)

    return barcode_frame


def decode_barcode(barcode_frame, return_frame=False):
    """
    This function decodes a barcode from the provided frame.

    Parameters:
    barcode_frame (np.array): The image frame to decode barcode from.
    return_frame (bool): Whether to return the frame with the decoded barcode drawn on it.

    Returns:
    tuple: The decoded barcode data and the frame with the barcode drawn on it, or None if no barcode is found.
    """
    decoded_barcode = None
    preprocessed_barcode_frame = preprocess_barcode(np.copy(barcode_frame))

    custom_config = r'-c tessedit_char_whitelist=1234567890 --psm 6'
    # Try out both the preprocessed and non-processed frames
    for frame in [preprocessed_barcode_frame, barcode_frame]:
        results = decode(frame, symbols=[ZBarSymbol.EAN13])

        if len(results) > 0:
            decoded_barcode = results[0].data.decode('utf-8')
            break

        text = pytesseract.image_to_string(frame, config=custom_config)
        # check if text is at least 13 digits, as expected for an EAN-13 Barcode
        match = re.search("\d{13}", text)
        if match:
            decoded_barcode = match.group().strip()
        else:
            decoded_barcode = None

    # Here the barcode can be checked against the products
    if decoded_barcode is not None:
        is_barcode_decoded = True
    else:
        is_barcode_decoded = False

    if return_frame is not False:
        # Write barcode onto image frame if barcode present is present
        if decoded_barcode is not None:
            preprocessed_barcode_frame = annotate_barcode_image(preprocessed_barcode_frame,decoded_barcode)

        # Join preprocessed barcode frame and raw frame vertically
        barcode_frame = cv2.vconcat([preprocessed_barcode_frame, barcode_frame])
    else:
        barcode_frame = None

    return is_barcode_decoded, barcode_frame
