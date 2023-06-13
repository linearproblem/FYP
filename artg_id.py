import cv2
import re
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gui_window import FONT_PATH

artg_id_to_product = {
    '390964': 'Kobalt (Hospital Grade Disinfectant)',
    '388590': 'Peracetic Acid (High-level Sterilant)',
    '387573': 'Bactol 90% (Surgical Handrub)',
    '354053': 'Medical Wipe (Device Cleaner)',
    '353791': 'Dental Stain (Caries Detector)',
    '353370': 'Medical Disinfectant (Device Cleanse)',
    '353333': 'Bactex (Hospital Grade Concentrate)',
    '345050': 'Bactol 2% CHG (Antiseptic Solution)',
    '335834': 'Bactol Clear (Disinfectant Gel)',
    '323850': 'Lubricant',
    '285421': 'Detergent',
    '284006': 'Glutaraldehyde (Sterilant)',
    '274095': 'Peracetic Acid (Sterilant 2)',
    '257360': 'Device Disinfectant (Second Cleanse)',
    '228023': 'Saniclear (70% Alcohol Gel)',
    '206575': 'OPA Disinfectant (Device Cleanse)',
    '206283': 'OPA Test Strip (Disinfectant Monitor)',
    '201558': 'Glutaral Test Strip (Sterilant Monitor)',
    '196551': 'Detergent 2',
    '187159': 'Detergent 3',
    '182622': 'Detergent 4',
    '155397': 'Bactol Blue (70% Topical Solution)',
    '135545': 'Glutaral (Sterilant 3)',
    '135544': 'Device Disinfectant (Third Cleanse)',
    '127945': 'Unclassified Product',
    '125529': 'Detergent 5',
    '125528': 'Detergent 6',
    '125527': 'Detergent 7',
    '73602': 'Sansol (Hospital Grade Disinfectant)',
    '69000': 'Viraclean (Industrial Disinfectant)',
}


def text_preprocessing(img_bgr, scale_factor=2, blur_value=(5, 5)):
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

    l_channel_adjusted = cv2.addWeighted(l_channel, 1.5, l_channel_mask.astype(l_channel.dtype), 0.01, 0.0)

    # Replace the Luminance channel in the original HSL image with the adjusted L channel
    # This is effectively the greyscale channel but this allows the result to be viewed in colour
    img_hsl[:, :, 1] = l_channel_adjusted.astype('uint8')

    # Convert the HSL image back to BGR
    img_processed = cv2.cvtColor(img_hsl, cv2.COLOR_HLS2BGR)

    return img_processed


def detect_artg_id(frame, return_frame=False):
    if frame is None:
        return None, None

    # Normalised co-ordinate values, suitable for 500mL Square bottles (only ones given with artg_id)
    x0 = 0.15
    x1 = 0.50
    y0 = 0.85  # Made a bit taller to have room to fit the text
    y1 = 0.95

    # Convert normalised coordinates to pixel values
    y0_px = int(y0 * frame.shape[0])
    y1_px = int(y1 * frame.shape[0])
    x0_px = int(x0 * frame.shape[1])
    x1_px = int(x1 * frame.shape[1])

    # Crop the image
    artg_frame = frame[y0_px:y1_px, x0_px:x1_px]

    # If frame is nothing after cropping
    if artg_frame is None:
        return None, None

    preprocessed_artg_frame = text_preprocessing(np.copy(artg_frame))

    # Perform OCR
    for frame in [preprocessed_artg_frame, artg_frame]:
        custom_config = r'-c tessedit_char_whitelist=AUSTR:1234567890/ --psm 6'
        extracted_text = pytesseract.image_to_string(frame, config=custom_config)

        # Check with regular expression
        number_match = re.search(r'AUSTR:(\d{6})', extracted_text)
        if number_match is not None:
            break
    if number_match is not None:
        artg_id = number_match.group(1)
        text = f"AUST R:{artg_id}"
        product_name = artg_id_to_product.get(artg_id, "Product not found")
        artg_found = True
    else:
        artg_found = False

    if return_frame:
        if artg_found:
            # Convert to PIL Image
            artg_frame_pil = Image.fromarray(cv2.cvtColor(preprocessed_artg_frame, cv2.COLOR_BGR2RGB))

            # Create a draw object
            draw = ImageDraw.Draw(artg_frame_pil)
            font_size = int(artg_frame_pil.height * 0.12)  # Scale font size to be 10% of image height
            font = ImageFont.truetype(FONT_PATH, font_size)

            # The starting point for the text, position text in top 10% of the image
            x = int(artg_frame_pil.width * 0.01)  # Start at left corner
            y = int(artg_frame_pil.height * 0.01)

            # Write barcode number to image
            text_width, text_height = draw.textsize(text, font=font)
            outline_color = "white"
            for adj in [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]:
                draw.text(adj, text, font=font, fill=outline_color)
            draw.text((x, y), text, font=font, fill="magenta")

            # Write product name below barcode, on a new line
            y += text_height
            font_size = int(artg_frame_pil.height * 0.08)  # These names can be really long
            font = ImageFont.truetype(FONT_PATH, font_size)
            for adj in [(x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)]:
                draw.text(adj, product_name, font=font, fill=outline_color)
            draw.text((x, y), product_name, font=font, fill="purple")  # change "purple" to any desired color

            # Convert back to OpenCV format
            preprocessed_artg_frame = cv2.cvtColor(np.array(artg_frame_pil), cv2.COLOR_RGB2BGR)

        frame = np.vstack([preprocessed_artg_frame, artg_frame])

    return artg_found, None if not return_frame else frame
