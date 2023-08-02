# Quality Checks

This is a basic summary of how the quality checks work.

## Barcode Decoder

The barcode image is pre-processed before it and the unprocessed barcode is fed into the pyzbar library to decode the barcode. During testing the processed image only occasionally worked better than the unprocessed image and sometimes it was worse, so processing may be unnecessary.

It also uses OCR (tesseract) to try and find the numbers below the barcode, which can work in a situation where the barcode cannot be decoded from the image.

## Bottle Orientation

Looks at the height/width of the bottle detection box and ensures the bottle is taller than it is high.

## Fill Level

Checks the fill level using gradient of the greyscale image. I have also tested this with individual colours with success. I also had some success with localised contrast thresholding for clear bottles but this was very much affected by lighting conditions. Also the angled piece near the lid affected the results for both coloured and clear liquids.

## ARTG ID

The barcode is assumed to be on the opposite side of the bottle to the ARTG ID, so if the barcode is detected by the rear camera, the cropped frame from the front camera is sent to this function. It is then further cropped using the known location of the ARTG ID relative to the bottle. This is then fed into tesseract OCR and compared against the list of known ARTG IDs.

## Bottle Cap

Uses canny thresholding and other pre-processing techniques to find the edges of the bottle and bottle lid. It then compares the distance between the known edges to ensure they are within a threshold range.

## Batch Information

Doesn't work reliably with tesseract OCR. I have had some better results using EasyOCR as well as Paddle OCR python libraries. The folder `\saved_images\ocr_test` has some images that I tested, a single line of text is easier than the double line to decipher.

## Label Straightness

This is just an example I had for demonstration day. But the plan was to incorporate the edge detection similar to the one used for the fill level to find more accurate locations for the edge of the bottle and to compare the edges to features near the top and bottom of the label on each side. Features such as: text (found using OCR) and barcode (using object detection), but the flammable warning and the Whiteley logo could also be used.
