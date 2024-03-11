# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

# Set initial parameters
learning_rate = 1e-4  # Learning rate for the optimizer
num_epochs = 10      # Total number of epochs for training
batch_size = 32      # Batch size for training and evaluation

# Specify dataset directory and categories
dataset_dir = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
label_categories = ["with_mask", "without_mask"]

# Load and preprocess the dataset
print("[INFO] loading images...")
all_data = []    # List to store all image data
all_labels = []  # List to store all labels

# Read images and labels from directory
for label in label_categories:
    folder_path = os.path.join(dataset_dir, label)
    for image_file in os.listdir(folder_path):
    	image_path = os.path.join(folder_path, image_file)
    	image = load_img(image_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	all_data.append(image)
    	all_labels.append(label)

# Encode labels into one-hot format
label_encoder = LabelBinarizer()
all_labels = label_encoder.fit_transform(all_labels)
all_labels = to_categorical(all_labels)

# Convert lists to numpy arrays
all_data = np.array(all_data, dtype="float32")
all_labels = np.array(all_labels)

# Split the dataset into training and testing sets
(train_images, test_images, train_labels, test_labels) = train_test_split(all_data, all_labels, test_size=0.20, stratify=all_labels, random_state=42)

# Initialize data augmentation for training
data_augmentation = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Load the MobileNetV2 model, without the top layer
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Build the new head for our model
new_head = base_model.output
new_head = AveragePooling2D(pool_size=(7, 7))(new_head)
new_head = Flatten(name="flatten")(new_head)
new_head = Dense(128, activation="relu")(new_head)
new_head = Dropout(0.5)(new_head)  # 50% dropout rate
new_head = Dense(2, activation="softmax")(new_head)

# Combine the base model with the new head
model = Model(inputs=base_model.input, outputs=new_head)

# Freeze the layers of the base model
for layer in base_model.layers:
	layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
optimizer = Adam(lr=learning_rate, decay=learning_rate / num_epochs)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
print("[INFO] training model...")
history = model.fit(
	data_augmentation.flow(train_images, train_labels, batch_size=batch_size),
	steps_per_epoch=len(train_images) // batch_size,
	validation_data=(test_images, test_labels),
	validation_steps=len(test_images) // batch_size,
	epochs=num_epochs)

# Evaluate the model on the testing set
print("[INFO] evaluating model...")
predictions = model.predict(test_images, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)
print(classification_report(test_labels.argmax(axis=1), predictions, target_names=label_encoder.classes_))

# Save the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# Plot the training and validation loss and accuracy
num_epochs_range = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs_range), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs_range), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs_range), history.history
