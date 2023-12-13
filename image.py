import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load a pre-trained model (e.g., MobileNetV2)
model = keras.applications.MobileNetV2(weights="imagenet")

# Load and preprocess the image
image_path = "2.jpg"
img = Image.open(image_path)
img = img.resize((224, 224))  # Resize the image to match the input size expected by the model
img = keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = keras.applications.mobilenet_v2.preprocess_input(img)

# Perform image classification
predictions = model.predict(img)

# Decode and display the top-5 predicted classes
decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

print("Top predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# Close the TensorFlow session
tf.keras.backend.clear_session()
