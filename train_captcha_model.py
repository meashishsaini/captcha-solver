import os
import tensorflow as tf
from tensorflow import keras
from imutils import paths
import cv2
import pickle
from helpers import resize_to_fit
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

LETTER_IMAGES_FOLDER = 'extracted_letter_images'
DIGIT_SUB = 'digit'
UPPER_SUB = 'upper'
LOWER_SUB = 'lower'

MODEL_FILENAME = "captcha_models/captcha_model"
MODEL_LABELS_FILENAME = "captcha_models/captcha_model_labels.dat"

# initialize the data and labels
data = []
labels = []

# loop over the input digits images
for image_file in paths.list_images(os.path.join(LETTER_IMAGES_FOLDER, DIGIT_SUB)):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Add a third channel dimension to the image to make Keras happy
	image = np.expand_dims(image, axis=2)

	# Grab the name of the letter based on the folder it was in
	label = image_file.split(os.path.sep)[-2]

	# Add the letter image and it's label to our training data
	data.append(image)
	labels.append(label)

# loop over the input upper case images
for image_file in paths.list_images(os.path.join(LETTER_IMAGES_FOLDER, UPPER_SUB)):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Add a third channel dimension to the image to make Keras happy
	image = np.expand_dims(image, axis=2)

	# Grab the name of the letter based on the folder it was in
	label = image_file.split(os.path.sep)[-2]

	# Add the letter image and it's label to our training data
	data.append(image)
	labels.append(label)

# loop over the input lower case images
for image_file in paths.list_images(os.path.join(LETTER_IMAGES_FOLDER, LOWER_SUB)):
	# Load the image and convert it to grayscale
	image = cv2.imread(image_file)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Add a third channel dimension to the image to make Keras happy
	image = np.expand_dims(image, axis=2)

	# Grab the name of the letter based on the folder it was in
	label = image_file.split(os.path.sep)[-2]

	# Add the letter image and it's label to our training data
	data.append(image)
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters and digits) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
	pickle.dump(lb, f)

model = keras.Sequential()

# First convolutional layer with max pooling
model.add(tf.keras.layers.Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(tf.keras.layers.Conv2D(50, (5, 5), padding="same", activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 250 nodes
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(250, activation=tf.nn.relu))

# Output layer with 53 nodes (one for each possible letter/number we predict)
# Upper 24 and lower 20 english alphabets and 9 digits
model.add(keras.layers.Dense(53, activation=tf.nn.softmax))


# Ask Keras to build the TensorFlow model behind the scenes
model.compile(optimizer=keras.optimizers.Adam(), 
				loss='categorical_crossentropy',
				metrics=['accuracy'])

# Print summary
model.summary()

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=53, epochs=5, verbose=1)

# # Evaluate the accuracy of the model
test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test accuracy:', test_acc)

# # Save the trained model to disk
model.save(MODEL_FILENAME)
