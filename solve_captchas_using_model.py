import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from helpers import resize_to_fit
from helpers import get_captcha_url
import numpy as np
import imutils
import cv2
import pickle
import requests
import cv2

MODEL_FILENAME = "captcha_models/captcha_model"
MODEL_LABELS_FILENAME = "captcha_models/captcha_model_labels.dat"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
	lb = pickle.load(f)

# Load the trained neural network
model = keras.models.load_model(MODEL_FILENAME)

# Grab captcha from the website to try to solve it
captcha_image_file = requests.get(get_captcha_url())

if captcha_image_file.status_code == requests.codes.OK: #pylint: disable=E1101

	# Load the image and convert it to grayscale
	image = np.asarray(bytearray(captcha_image_file.content), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

	# Increase image size for better recognition
	image = cv2.resize(image, None, fx=3.3, fy=3.3, interpolation=cv2.INTER_CUBIC)
	temp_img = image.copy()
	# Remove noise from image
	image = cv2.fastNlMeansDenoising(src=image, h=40)
	
	# Convert it to PIL image to work with pixels directly
	#img = Image.fromarray(image)
	#img = img.convert("RGBA")
	#pixdata = img.load()

	# Make the letters bolder for easier recognition
	height, width = image.shape
	for y in range(width):
		for x in range(height):
			if image[x, y] < 90:
				image[x, y] = 0		
	for y in range(width):
		for x in range(height):
			if image[x, y] < 120: #orig 136
				image[x, y] = 0		
	for y in range(width):
		for x in range(height):
			if image[x, y] > 0:
				image[x, y] = 255
	
	#image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

	# threshold the image (convert it to pure black and white)
	thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# find the contours (continuous blobs of pixels) the image
	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# # Hack for compatibility with different OpenCV versions
	contours = contours[0] #if imutils.is_cv2() else contours[1]

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
		if w > 79:
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
	if len(letter_image_regions) == 5:

		# Sort the detected letter images based on the x coordinate to make sure
		# we are processing them from left-to-right so we match the right image
		# with the right letter
		letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

		# Create an output image and a list to hold our predicted letters
		output = cv2.merge([temp_img] * 3)
		predictions = []

		# loop over the letters
		for letter_bounding_box in letter_image_regions:
			# Grab the coordinates of the letter in the image
			x, y, w, h = letter_bounding_box

			# For saving original height to detect difference b/w upper and lower characters
			(x2, y2, w2, h2) = cv2.boundingRect(image)

			# Extract the letter from the original image with a 2-pixel margin around the edge
			letter_image = image[0:y + h2 + 2, x - 2:x + w + 2]

			# Re-size the letter image to 20x20 pixels to match training data
			letter_image = resize_to_fit(letter_image, 20, 20)

			# Turn the single image into a 4d list of images to make Keras happy
			letter_image = np.expand_dims(letter_image, axis=2)
			letter_image = np.expand_dims(letter_image, axis=0)

			# Ask the neural network to make a prediction
			prediction = model.predict(letter_image)

			# Convert the one-hot-encoded prediction back to a normal letter
			letter = lb.inverse_transform(prediction)[0]
			predictions.append(letter)

			# draw the prediction on the output image
			cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (128, 0, 255), 1)
			cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 0, 255), 2)

		# Print the captcha's text
		captcha_text = "".join(predictions)
		print("CAPTCHA text is: {}".format(captcha_text))

		# Show the annotated image
		cv2.imshow("Output", output)
		cv2.waitKey()
	else:
		print("More or less letters found: " + str(len(letter_image_regions)))