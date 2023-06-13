import cv2
import re
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gui_window import FONT_PATH

def text_preprocessing(img_bgr, scale_factor=3, blur_value=(5, 5)):
    """
    Function to pre-process image for text detection.
    Parameters:
        img_bgr (str): The path to the image file
        scale_factor (float): The factor by which to scale the image
        blur_value (tuple): The kernel size for the Gaussian blur
    Returns:
        img_processed (np.array): The processed image array
    """

    # Convert the BGR image to HSL
    img_hsl = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

    # Extract the L channel
    l_channel = img_hsl[:, :, 1]

    # Scale the L channel
    l_channel_scaled = cv2.resize(l_channel, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to the L channel
    l_channel_blurred = cv2.GaussianBlur(l_channel_scaled, blur_value, 0)

    # Apply Adaptive Thresholding to the L channel
    l_channel_mask = cv2.adaptiveThreshold(np.copy(l_channel_blurred), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY,
                                           5, 2)

    # Scale the mask back to original L channel size
    l_channel_mask = cv2.resize(l_channel_mask, (l_channel.shape[1], l_channel.shape[0])) // 255.0

    # Adjust the luminance of the original L channel using the mask

    l_channel_adjusted = cv2.addWeighted(l_channel, 0.5, l_channel_mask.astype(l_channel.dtype), 0.01, 0.0)

    # Replace the Luminance channel in the original HSL image with the adjusted L channel
    # This is effectively the greyscale channel but this allows the result to be viewed in colour
    img_hsl[:, :, 1] = l_channel_adjusted.astype('uint8')

    # Convert the HSL image back to BGR
    img_processed = cv2.cvtColor(img_hsl, cv2.COLOR_HLS2BGR)

    return img_processed

def detect_batch_expiry_and_best_before(frame, return_frame=False):
    if frame is None:
        return None, None

    # Normalised co-ordinate values, suitable for 500mL Square bottles (only ones given with batch numbers, expiry and best before dates)
    x0 = 0.08
    x1 = 0.96
    y0 = 0.20  # Made a bit taller to have room to fit the text
    y1 = 0.30

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

    preprocessed_info_frame = text_preprocessing(np.copy(info_frame))

    # Perform OCR
    for frame in [preprocessed_info_frame, info_frame]:
        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ:1234567890/ --psm 6'
        extracted_text = pytesseract.image_to_string(frame, config=custom_config)
        print(extracted_text)
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

    return info_found, None if not return_frame else frame
