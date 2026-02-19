import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys

# ===============================
# Load SavedModel
# ===============================
model = tf.saved_model.load("plant_model_tf")

# Get serving function
infer = model.signatures["serve"]

# ===============================
# Load class names
# ===============================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ===============================
# Preprocess image
# ===============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# ===============================
# Predict
# ===============================
if len(sys.argv) < 2:
    print("Usage: python predict.py image.jpg")
    sys.exit()

image_path = sys.argv[1]

input_tensor = preprocess_image(image_path)

# Run inference
output = infer(input_tensor)

# Get predictions
predictions = list(output.values())[0].numpy()
predicted_index = np.argmax(predictions)
confidence = float(np.max(predictions))

print("\n✅ Prediction:", class_names[predicted_index])
print("📊 Confidence: {:.2f}%".format(confidence * 100))
