import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# ==========================================
# LOAD TENSORFLOW MODEL
# ==========================================

MODEL_PATH = "plant_model_tf"

try:
    loaded_model = tf.saved_model.load(MODEL_PATH)
    infer = loaded_model.signatures["serving_default"]
    print("✅ TensorFlow model loaded")
except Exception as e:
    print("❌ Model Load Failed:", e)
    infer = None


# ==========================================
# GEMINI CONFIG
# ==========================================

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("❌ GEMINI_API_KEY not found")
    client = None
else:
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        print("✅ Gemini API Loaded")
    except Exception as e:
        print("❌ Gemini Client Error:", e)
        client = None


# ==========================================
# LOAD CLASS DATA
# ==========================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("remedies.json", "r", encoding="utf-8") as f:
    remedies = json.load(f)


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# LIST AVAILABLE GEMINI MODELS
# ==========================================

@app.route("/models")
def list_models():
    if client is None:
        return jsonify({"error": "Gemini not configured"})
    try:
        models = client.models.list()
        return jsonify([m.name for m in models])
    except Exception as e:
        return jsonify({"error": str(e)})


# ==========================================
# OFFLINE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if infer is None:
            return jsonify({"error": "Model not loaded"})

        file = request.files["file"]

        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        upload_path = os.path.join(upload_folder, file.filename)
        file.save(upload_path)

        img = tf.keras.utils.load_img(upload_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        predictions = infer(img_tensor)
        predictions = list(predictions.values())[0].numpy()

        idx = np.argmax(predictions)
        confidence = float(predictions[0][idx]) * 100
        predicted_class = class_names[idx]

        remedy = remedies.get(predicted_class)

        if not remedy:
            return jsonify({"error": "Remedy not found"})

        return jsonify({
            "diagnosis": remedy["title"],
            "confidence": round(confidence, 2),
            "steps_en": remedy["steps_en"],
            "steps_mr": remedy["steps_mr"]
        })

    except Exception as e:
        print("Offline Error:", str(e))
        return jsonify({"error": str(e)})


# ==========================================
# ONLINE GEMINI PREDICTION
# ==========================================

@app.route("/predict_online", methods=["POST"])
def predict_online():
    try:
        if client is None:
            return jsonify({"error": "Gemini not configured"})

        if infer is None:
            return jsonify({"error": "Model not loaded"})

        file = request.files["file"]

        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        upload_path = os.path.join(upload_folder, file.filename)
        file.save(upload_path)

        img = tf.keras.utils.load_img(upload_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        predictions = infer(img_tensor)
        predictions = list(predictions.values())[0].numpy()

        idx = np.argmax(predictions)
        predicted_class = class_names[idx]

        prompt = f"""
A crop leaf disease has been detected: {predicted_class}.

Provide:
1. Detailed explanation.
2. Treatment steps in English.
3. Treatment steps in Marathi.
4. Organic solutions.
"""

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )

            if not response.text:
                return jsonify({"error": "Empty Gemini response"})

            return jsonify({
                "diagnosis": predicted_class,
                "ai_response": response.text
            })

        except Exception as gemini_error:
            print("Gemini Error:", gemini_error)
            return jsonify({"error": "Gemini API failed"})

    except Exception as e:
        print("Online Error:", str(e))
        return jsonify({"error": str(e)})


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
