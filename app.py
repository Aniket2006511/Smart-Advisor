import os
import io
import re
import base64
import json
import numpy as np
import tensorflow as tf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image



load_dotenv()
app = Flask(__name__)

# =========================
# LOAD TENSORFLOW MODEL
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

OPENROUTER_MODELS = [
    # Qwen Vision - best accuracy for plant analysis
    "qwen/qwen2.5-vl-72b-instruct:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen/qwen3-vl-30b-a3b-thinking",
    # Meta Llama Vision
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-3.2-90b-vision-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    # Google
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    # Others
    "moonshotai/kimi-vl-a3b-thinking:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    # New high accuracy models
    "google/gemini-2.5-pro-exp-03-25:free",
    "bytedance-research/ui-tars-72b:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "openrouter/free",
]

if OPENROUTER_KEY:
    print("✅ OpenRouter Ready")
else:
    print("⚠️ OpenRouter not configured")

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
# WEATHER
# =========================
@app.route("/weather")
def get_weather():
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        city = "Pune"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()

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
# NORMALIZE AI RESPONSE
# =========================
def normalize_ai_text(ai_text):
    for label in ["CROP:", "DISEASE:", "CHEMICAL_EN:", "CHEMICAL_MR:", "ORGANIC_EN:", "ORGANIC_MR:"]:
        ai_text = ai_text.replace(label, f"\n{label}\n")
    ai_text = re.sub(r' - ', '\n- ', ai_text)
    return ai_text.strip()

# =========================
# ONLINE PREDICTION
# =========================
@app.route("/predict_online", methods=["POST"])
def predict_online():
    if not OPENROUTER_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not configured in .env"})

    try:
        file = request.files["file"]
        img_bytes = file.read()

        # Validate image
        try:
            Image.open(io.BytesIO(img_bytes)).verify()
        except Exception:
            return jsonify({"error": "Invalid image file"})

        # Compress to 512px for faster upload
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((512, 512))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        img_bytes = buffer.getvalue()
        mime_type = "image/jpeg"

        prompt = (
            "You are an expert plant pathologist and agricultural scientist with deep knowledge of crop diseases.\n"
            "Carefully examine this leaf image and identify the crop and disease with high accuracy.\n"
            "If unsure about crop, write 'Unknown'. If the plant is healthy, write 'Healthy'.\n\n"
            "Reply ONLY in this exact format with no extra text, no explanations, no markdown:\n\n"
            "CROP: [crop name]\n"
            "DISEASE: [disease name or Healthy]\n"
            "CHEMICAL_EN:\n"
            "- [EXACT chemical active ingredient name as scientifically registered (e.g. Mancozeb, Carbendazim, Propiconazole, Copper oxychloride) followed by dosage only, max 10 words total]\n"
            "- [EXACT chemical active ingredient name + dosage, max 10 words]\n"
            "- [EXACT chemical active ingredient name + dosage, max 10 words]\n"
            "- [EXACT chemical active ingredient name + dosage, max 10 words]\n"
            "CHEMICAL_MR:\n"
            "- [Copy the EXACT same chemical name from CHEMICAL_EN point 1 without any change or translation — only convert the dosage units to Marathi e.g. per litre → प्रति लिटर, per hectare → प्रति हेक्टर, gram → ग्रॅम, ml → मिली]\n"
            "- [Copy EXACT chemical name from CHEMICAL_EN point 2 + dosage units in Marathi only]\n"
            "- [Copy EXACT chemical name from CHEMICAL_EN point 3 + dosage units in Marathi only]\n"
            "- [Copy EXACT chemical name from CHEMICAL_EN point 4 + dosage units in Marathi only]\n"
            "STRICT RULE FOR CHEMICAL_MR: The chemical ingredient name must be 100% identical to CHEMICAL_EN. Do NOT translate, transliterate, or modify the chemical name in any way. Only the dosage units are converted to Marathi script.\n"
            "ORGANIC_EN:\n"
            "- [detailed organic remedy - how to prepare it, how to apply it, how often, 2-3 sentences]\n"
            "- [detailed organic remedy - how to prepare it, how to apply it, how often, 2-3 sentences]\n"
            "- [detailed organic remedy - how to prepare it, how to apply it, how often, 2-3 sentences]\n"
            "- [detailed organic remedy - how to prepare it, how to apply it, how often, 2-3 sentences]\n"
            "- [detailed organic remedy - how to prepare it, how to apply it, how often, 2-3 sentences]\n"
            "ORGANIC_MR:\n"
            "- [वरील पहिला सेंद्रिय उपाय संपूर्ण मराठीत - तयारी, वापर आणि वेळ सांगा]\n"
            "- [वरील दुसरा सेंद्रिय उपाय संपूर्ण मराठीत]\n"
            "- [वरील तिसरा सेंद्रिय उपाय संपूर्ण मराठीत]\n"
            "- [वरील चौथा सेंद्रिय उपाय संपूर्ण मराठीत]\n"
            "- [वरील पाचवा सेंद्रिय उपाय संपूर्ण मराठीत]\n"
            "FINAL RULES:\n"
            "- Chemical points: exactly 4, short and specific with correct dosage.\n"
            "- Organic points: exactly 5, each must be 2-3 detailed sentences.\n"
            "- Marathi must be proper Marathi script only, no Roman/English script in Marathi sections except chemical names.\n"
            "- Do not add any extra text, headings, or explanations outside the format above.\n"
            "Also add this at the very end:\n"
            "CONFIDENCE: [0-100% how confident you are this diagnosis is correct]"
        )

        image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        def try_model(model):
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
                    "max_tokens": 1200,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                },
                timeout=30
            )
            result = response.json()
            if "error" in result:
                raise Exception(result["error"].get("message", "Unknown error"))
            ai_text = result["choices"][0]["message"]["content"]
            if not ai_text or ai_text.strip() == "":
                raise Exception("Empty response")
            return model, ai_text

        # Try models in parallel
        last_error = None
        winner_model = None
        winner_text = None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(try_model, m): m for m in OPENROUTER_MODELS[:8]}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    winner_model, winner_text = future.result()
                    print(f"✅ Parallel success: {winner_model}")
                    for f in futures:
                        f.cancel()
                    break
                except Exception as e:
                    print(f"❌ {model} failed: {e}")
                    last_error = str(e)

        if not winner_text:
            for model in OPENROUTER_MODELS[8:]:
                try:
                    winner_model, winner_text = try_model(model)
                    print(f"✅ Sequential success: {winner_model}")
                    break
                except Exception as e:
                    print(f"❌ {model} failed: {e}")
                    last_error = str(e)

        if winner_text:
            import re as re2
            conf_match = re2.search(r'CONFIDENCE:\s*(\d+(?:\.\d+)?%?)', winner_text, re2.IGNORECASE)
            confidence_str = conf_match.group(1) if conf_match else "N/A"
            # Remove confidence line from ai_response display
            clean_text = re2.sub(r'\nCONFIDENCE:.*', '', winner_text).strip()
            return jsonify({
                "diagnosis": "AI Vision Analysis",
                "ai_response": normalize_ai_text(clean_text),
                "model_used": winner_model,
                "ai_confidence": confidence_str
            })

        return jsonify({"error": f"All AI models failed. Last error: {last_error}"})

    except Exception as e:
        print("Online Prediction Error:", e)
        return jsonify({"error": str(e)})

# =========================
# MARKET PAGE
# =========================
@app.route("/market")
def market():
    chemicals = request.args.get("chemicals", "")
    chemical_list = chemicals.split("|") if chemicals else []
    return render_template("market.html", chemicals=chemical_list)


# =========================
# EXPERIENCE PAGE
# =========================
@app.route("/experience")
def experience():
    return render_template("experience.html")


# =========================
# PLAN PAGE
# =========================
@app.route("/plan")
def plan():
    return render_template("plan.html")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
