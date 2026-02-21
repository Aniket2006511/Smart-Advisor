import os
import io
import base64
import json
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

app = Flask(__name__)

# =========================
# LOAD TENSORFLOW MODEL (OFFLINE)
# =========================
MODEL_PATH = "plant_model_tf"
infer = None

try:
    loaded_model = tf.saved_model.load(MODEL_PATH)
    infer = loaded_model.signatures["serving_default"]
    print("✅ TensorFlow model loaded")
except Exception as e:
    print("❌ Model Load Failed:", e)

# =========================
# OPENROUTER CONFIG
# =========================
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Free vision-capable models on OpenRouter (tried in priority order)
# openrouter/free = auto-picks any available free vision model (best for reliability)
OPENROUTER_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct:free",             # Best free vision - 72B, very accurate
    "qwen/qwen2.5-vl-32b-instruct:free",             # Qwen vision 32B fallback
    "moonshotai/kimi-vl-a3b-thinking:free",          # Kimi VL - good vision
    "google/gemma-3-27b-it:free",                    # Gemma 27B vision
    "meta-llama/llama-3.2-11b-vision-instruct:free", # Llama vision
    "mistralai/mistral-small-3.1-24b-instruct:free", # Mistral vision
    "nvidia/nemotron-nano-12b-v2-vl:free",           # NVIDIA vision
    "openrouter/free",                                # Final fallback - auto pick
]

if OPENROUTER_KEY:
    print("✅ OpenRouter Ready")
else:
    print("❌ OPENROUTER_API_KEY missing")

# =========================
# LOAD CLASS + REMEDY DATA
# =========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("remedies.json", "r", encoding="utf-8") as f:
    remedies = json.load(f)

print("📌 Number of classes:", len(class_names))

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# WEATHER (PUNE FIXED)
# =========================
@app.route("/weather")
def get_weather():
    try:
        api_key = "70179288ce4e3962226e3efd4e042cdc"
        city = "Pune"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        print("FULL WEATHER RESPONSE:", data)

        if "main" not in data:
            return jsonify({"error": "Weather API error"})

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]

        if temp > 35:
            advice = "Too hot for spraying"
        elif temp < 15:
            advice = "Too cold for spraying"
        else:
            advice = "Good time to spray"

        return jsonify({"temp": temp, "condition": condition.title(), "advice": advice})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# OFFLINE PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if infer is None:
        return jsonify({"error": "Model not loaded"})

    try:
        file = request.files["file"]
    except KeyError:
        return jsonify({"error": "No file provided"})

    try:
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        upload_path = os.path.join(upload_folder, file.filename)
        file.save(upload_path)

        img = tf.keras.utils.load_img(upload_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        predictions = infer(img_tensor)
        pred_array = list(predictions.values())[0].numpy()

        if len(pred_array.shape) == 2:
            pred_array = pred_array[0]

        idx = int(np.argmax(pred_array))
        confidence = float(pred_array[idx]) * 100

        predicted_class = class_names[idx]
        remedy = remedies.get(predicted_class)

        if not remedy:
            return jsonify({"error": "Remedy not found"})

        return jsonify({
            "diagnosis": remedy["title"],
            "confidence": round(confidence, 2),
            "steps_en": remedy["steps_en"],
            "steps_mr": remedy["steps_mr"],
            "organic_en": remedy.get("organic_en", []),
            "organic_mr": remedy.get("organic_mr", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# ONLINE PREDICTION (OPENROUTER)
# =========================
@app.route("/predict_online", methods=["POST"])
def predict_online():
    if not OPENROUTER_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not configured in .env"})

    try:
        file = request.files["file"]
        img_bytes = file.read()
        mime_type = file.mimetype or "image/jpeg"

        try:
            Image.open(io.BytesIO(img_bytes)).verify()
        except Exception:
            return jsonify({"error": "Invalid image file"})

        image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        last_error = None
        for model in OPENROUTER_MODELS:
            try:
                print(f"Trying OpenRouter model: {model}")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://smart-crop-advisor.app",
                        "X-Title": "Smart Crop Advisor",
                    },
                    json={
                        "model": model,
                        "max_tokens": 600,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                                    },
                                    {
                                        "type": "text",
                                        "text": (
                                            "You are an expert plant pathologist. Carefully examine this leaf image.\n"
                                            "Identify the exact crop and any disease visible. Be precise - do not guess.\n"
                                            "If unsure about crop, write 'Unknown'. If leaf is healthy, write 'Healthy'.\n\n"
                                            "Reply ONLY in this exact format:\n\n"
                                            "CROP: [exact crop name]\n"
                                            "DISEASE: [exact disease name or Healthy]\n"
                                            "CHEMICAL_EN:\n"
                                            "- [specific treatment point]\n"
                                            "- [specific treatment point]\n"
                                            "- [specific treatment point]\n"
                                            "- [specific treatment point]\n"
                                            "- [specific treatment point]\n"
                                            "CHEMICAL_MR:\n"
                                            "- [above 5 points in proper Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "ORGANIC_EN:\n"
                                            "- [organic remedy point]\n"
                                            "- [organic remedy point]\n"
                                            "- [organic remedy point]\n"
                                            "- [organic remedy point]\n"
                                            "- [organic remedy point]\n"
                                            "ORGANIC_MR:\n"
                                            "- [above 5 points in proper Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "- [Marathi]\n"
                                            "Write exactly 5 points per section. No extra text."
                                        )
                                    }
                                ]
                            }
                        ],
                    },
                    timeout=30
                )

                result = response.json()
                if "error" in result:
                    raise Exception(result["error"].get("message", "Unknown error"))

                ai_text = result["choices"][0]["message"]["content"]

                # If model returned empty, try next one
                if not ai_text or ai_text.strip() == "":
                    raise Exception("Model returned empty response")

                # Force each section label onto its own line
                for label in ["CROP:", "DISEASE:", "CHEMICAL_EN:", "CHEMICAL_MR:", "ORGANIC_EN:", "ORGANIC_MR:"]:
                    ai_text = ai_text.replace(label, f"\n{label}\n")
                # Force each bullet onto its own line
                ai_text = ai_text.replace(" - ", "\n- ")
                ai_text = ai_text.strip()

                # Force each label and bullet onto its own line
                for label in ["CROP:", "DISEASE:", "CHEMICAL_EN:", "CHEMICAL_MR:", "ORGANIC_EN:", "ORGANIC_MR:"]:
                    ai_text = ai_text.replace(label, f"\n{label}\n")
                # Split bullets that are joined inline: "- x - y" → each on own line
                import re
                ai_text = re.sub(r'\s+-\s+', '\n- ', ai_text)
                ai_text = ai_text.strip()

                # Normalize: force section labels onto their own lines
                for label in ["CROP:", "DISEASE:", "CHEMICAL_EN:", "CHEMICAL_MR:", "ORGANIC_EN:", "ORGANIC_MR:"]:
                    ai_text = ai_text.replace(label, f"\n{label}\n")
                ai_text = ai_text.replace(" - ", "\n- ")

                print(f"✅ Success with: {model}")
                return jsonify({
                    "diagnosis": "AI Vision Analysis",
                    "ai_response": ai_text,
                    "model_used": model
                })

            except Exception as e:
                print(f"❌ {model} failed: {e}")
                last_error = str(e)
                continue

        return jsonify({"error": f"All AI models failed. Last error: {last_error}"})

    except Exception as e:
        print("OpenRouter Error:", e)
        return jsonify({"error": str(e)})

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)