# Import libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

def detect_and_predict(frame, detector, model_mask):
	# Get frame dimensions
	frame_height, frame_width = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(224, 224), mean=(104.0, 177.0, 123.0))

	# Detect faces
	detector.setInput(blob)
	detections = detector.forward()

	# Prepare lists for faces, locations, and predictions
	face_list, locs, predictions = [], [], []

	# Process each detection
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]  # Get confidence

		# Filter weak detections
		if confidence > 0.5:
			# Calculate bounding box
			box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
			startX, startY, endX, endY = box.astype("int")

			# Ensure box is within frame
			startX, startY = max(0, startX), max(0, startY)
			endX, endY = min(frame_width - 1, endX), min(frame_height - 1, endY)

			# Extract, convert, resize, and preprocess face
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Add to lists
			face_list.append(face)
			locs.append((startX, startY, endX, endY))

	# Predict mask if faces detected
	if len(face_list) > 0:
		faces_batch = np.array(face_list, dtype="float32")
		predictions = model_mask.predict(faces_batch, batch_size=32)

	# Return locations and predictions
	return locs, predictions

# Load face detector and mask model
path_prototxt = "face_detector/deploy.prototxt"
path_weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNet(path_prototxt, path_weights)
model_mask = load_model("mask_detector.model")

# Start video stream
print("Starting video stream...")
stream = VideoStream(src=0).start()

# Read and process frames
while True:
	frame = stream.read()
	frame = imutils.resize(frame, width=400)  # Resize frame

	# Detect faces and predict mask usage
	locs, preds = detect_and_predict(frame, detector, model_mask)

	# Display results
	for (box, pred) in zip(locs, preds):
		startX, startY, endX, endY = box
		mask, no_mask = pred
		label = "Mask" if mask > no_mask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, no_mask) * 100)
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):  # Quit on 'q' key
		break

# Clean up
cv2.destroyAllWindows()
stream.stop()
