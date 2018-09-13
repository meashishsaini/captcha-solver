import os
import os.path
import cv2
import glob
from PIL import Image
from helpers import resize_to_fit
import imutils

CAPTCHA_IMAGE_FOLDER = "transformed_captchas"
OUTPUT_FOLDER = "extracted_letter_images"
DIGIT_SUB = '/digit'
UPPER_SUB = '/upper'
LOWER_SUB = '/lower'

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Extract the base filename as the captcha text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 0, 0, 8, 8, cv2.BORDER_REPLICATE)
    
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # If countour width and height is less than expected than skip it
        if w < 20  or h < 20:
            continue
        # Compare the width of the contour to detect letters that are conjoined into one chunk
        if w >= 79:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 5 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 5:
        print("More or less letters found " + str(len(letter_image_regions)))
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # For saving original height to detect difference b/w upper and lower characters
        (x2, y2, w2, h2) = cv2.boundingRect(gray)

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[0:y + h2 + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        if letter_text.isdigit():
            t = OUTPUT_FOLDER + DIGIT_SUB
        elif letter_text.isupper():
            t = OUTPUT_FOLDER + UPPER_SUB
        else:
            t = OUTPUT_FOLDER + LOWER_SUB
        save_path = os.path.join(t, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        letter_image = resize_to_fit(letter_image, 20, 20)
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1