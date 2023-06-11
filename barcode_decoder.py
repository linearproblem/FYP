import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\OCR\Tesseract-OCR\tesseract.exe'  # Required for Windows


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
    results = decode(barcode_frame, symbols=[ZBarSymbol.EAN13])

    if len(results) == 0:
        # Use pytesseract OCR to check barcode as backup
        custom_config = r'-c tessedit_char_whitelist=1234567890 --psm 6'
        text = pytesseract.image_to_string(barcode_frame, config=custom_config)
        if text:
            decoded_barcode = text.strip()
    else:
        decoded_barcode = results[0].data.decode('utf-8')

    if return_frame is not False:
        # Write barcode onto image frame if barcode present is present
        if decoded_barcode is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            x = 120
            y = 20
            padding = 20
            text = decoded_barcode
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            box_top_left = (x - padding, y - padding)
            box_bottom_right = (x + text_width + padding, y + text_height + padding)
            cv2.rectangle(barcode_frame, box_top_left, box_bottom_right, (255, 255, 255), -1)
            cv2.putText(barcode_frame, text, (x, y), font, font_scale, (255, 0, 255), thickness)
    else:
        barcode_frame = None

    return decoded_barcode, barcode_frame
