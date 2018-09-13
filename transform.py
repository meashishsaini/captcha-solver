from PIL import Image
import numpy as np
import cv2
import os
import glob

CAPTCHA_DOWNLOAD_FOLDER = "saved_captchas"
OUTPUT_FOLDER = "transformed_captchas"

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_DOWNLOAD_FOLDER, "*"))
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
	print("Transforming image {}/{}".format(i + 1, len(captcha_image_files)))

	# Extract the base filename as the captcha text
	filename = os.path.basename(captcha_image_file)
	captcha_correct_text = os.path.splitext(filename)[0]

	cvimg = cv2.imread(captcha_image_file)
	cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
	cvimg = cv2.resize(cvimg, None, fx=3.3, fy=3.3, interpolation=cv2.INTER_CUBIC)
	cvimg = cv2.fastNlMeansDenoising(src=cvimg, h=40)

	# Make the letters bolder for easier recognition
	height, width = cvimg.shape
	for y in range(width):
		for x in range(height):
			if cvimg[x, y] < 90:
				cvimg[x, y] = 0		
	for y in range(width):
		for x in range(height):
			if cvimg[x, y] < 120: #orig 136
				cvimg[x, y] = 0		
	for y in range(width):
		for x in range(height):
			if cvimg[x, y] > 0:
				cvimg[x, y] = 255
	save_path = os.path.join(OUTPUT_FOLDER, captcha_correct_text + '.png')
	cv2.imwrite(save_path, cvimg)