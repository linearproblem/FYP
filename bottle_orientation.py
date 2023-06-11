from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def is_bottle_upright(bottle_frame, return_frame=False):
    # Check bottle orientation based on frame dimensions
    bottle_upright = bottle_frame.shape[0] > bottle_frame.shape[1]

    # Only generate status image if return_frame is True
    status_image = None
    if return_frame:
        # Create an image of 200px wide by 100px high
        status_image = Image.new('RGB', (200, 100))

        # Prepare to draw on the image
        draw = ImageDraw.Draw(status_image)

        # Set the message based on bottle orientation
        message = 'Upright' if bottle_upright else 'Fallen over'

        # Set the color based on bottle orientation
        color = 'green' if bottle_upright else 'red'

        # Load a font (this will depend on your system fonts)
        font = ImageFont.truetype("arial", size=30)

        # Center the text
        text_width, text_height = draw.textsize(message, font=font)
        position = ((200 - text_width) / 2, (100 - text_height) / 2)

        # Draw the text on the image
        draw.text(position, message, fill=color, font=font)

        # Convert PIL image to OpenCV image
        status_image = cv2.cvtColor(np.array(status_image), cv2.COLOR_RGB2BGR)

    return bottle_upright, status_image
