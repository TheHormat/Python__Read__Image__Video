import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load a pre-trained model (e.g., MobileNetV2)
model = keras.applications.MobileNetV2(weights="imagenet")

# Open the video file
cap = cv2.VideoCapture("1.mp4")

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to an image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))  # Resize the image to match the input size expected by the model
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.mobilenet_v2.preprocess_input(img)

    # Perform image classification
    predictions = model.predict(img)

    # Decode and display the top-5 predicted classes
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    # Display the results on the frame
    for i, (_, label, score) in enumerate(decoded_predictions):
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame with results
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the TensorFlow session
tf.keras.backend.clear_session()
