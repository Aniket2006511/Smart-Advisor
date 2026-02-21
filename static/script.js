let voiceEnabled = true;

let lastEnglishSteps = [];
let lastMarathiSteps = [];
let lastAIResponse = "";

// ===============================
// ANALYZE IMAGE (ONLINE + OFFLINE)
// ===============================
function analyzeImage() {

  const input = document.getElementById("imageInput");

  if (!input.files.length) {
    alert("Select an image first.");
    return;
  }

  const selectedMode = document.querySelector('input[name="mode"]:checked').value;
  const endpoint = selectedMode === "online" ? "/predict_online" : "/predict";

  const formData = new FormData();
  formData.append("file", input.files[0]);

  console.log("Sending to:", endpoint);

  fetch(endpoint, {
    method: "POST",
    body: formData,
  })
    .then(res => res.json())
    .then(data => {

      console.log("Server response:", data);

      if (data.error && selectedMode === "online") {
        console.log("Online failed. Switching to offline...");
        fallbackToOffline();
        return;
      }

      // Also fallback if online returned empty response
      if (selectedMode === "online" && (!data.ai_response || data.ai_response.trim() === "")) {
        console.log("Online returned empty. Switching to offline...");
        fallbackToOffline();
        return;
      }

      if (data.error) {
        alert(data.error);
        return;
      }

      if (selectedMode === "online") {
        showOnlineResult(data);
      } else {
        showOfflineResult(data);
      }

    })
    .catch(err => {
      console.log("Fetch error:", err);
      if (selectedMode === "online") {
        fallbackToOffline();
      } else {
        alert("Prediction failed.");
      }
    });
}

// ===============================
// FALLBACK TO OFFLINE
// ===============================
function fallbackToOffline() {
  // Re-build FormData fresh from the file input (stream was consumed)
  const input = document.getElementById("imageInput");
  if (!input.files.length) {
    alert("Offline prediction failed: no file available.");
    return;
  }

  const freshFormData = new FormData();
  freshFormData.append("file", input.files[0]);

  fetch("/predict", {
    method: "POST",
    body: freshFormData,
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      showOfflineResult(data);
    })
    .catch(() => {
      alert("Offline prediction failed.");
    });
}

// ===============================
// OFFLINE RESULT
// ===============================
function showOfflineResult(data) {

  document.getElementById("result-section").style.display = "block";

  document.getElementById("disease-name").innerHTML =
    "<b>Diagnosis:</b> " + data.diagnosis +
    "<br><b>Confidence:</b> " + data.confidence + "%";

  let stepsHTML = "<h4>🧪 Chemical Treatment (English):</h4><ul>";
  data.steps_en.forEach(step => {
    stepsHTML += "<li>" + step + "</li>";
  });
  stepsHTML += "</ul>";

  stepsHTML += "<h4>🧪 Chemical Treatment (मराठी):</h4><ul>";
  data.steps_mr.forEach(step => {
    stepsHTML += "<li>" + step + "</li>";
  });
  stepsHTML += "</ul>";

  // 🌿 ORGANIC SECTION
  if (data.organic_en && data.organic_en.length > 0) {

    stepsHTML += "<h4 style='color:green;'>🌿 Organic Treatment (English):</h4><ul>";
    data.organic_en.forEach(step => {
      stepsHTML += "<li>" + step + "</li>";
    });
    stepsHTML += "</ul>";

    stepsHTML += "<h4 style='color:green;'>🌿 सेंद्रिय उपाय (मराठी):</h4><ul>";
    data.organic_mr.forEach(step => {
      stepsHTML += "<li>" + step + "</li>";
    });
    stepsHTML += "</ul>";
  }

  document.getElementById("remedy-content").innerHTML = stepsHTML;

  // 🔥 STORE ALL FOR VOICE
  lastEnglishSteps = [...data.steps_en, ...(data.organic_en || [])];
  lastMarathiSteps = [...data.steps_mr, ...(data.organic_mr || [])];
  lastAIResponse = "";

  speakAdvice(lastEnglishSteps, lastMarathiSteps);
}

// ===============================
// ONLINE RESULT (OpenRouter)
// ===============================
function showOnlineResult(data) {

  document.getElementById("result-section").style.display = "block";

  const raw = data.ai_response || "";

  // Parse structured sections by label
  function extractSection(text, label) {
    const regex = new RegExp(label + ":\\s*\\n([\\s\\S]*?)(?=\\n[A-Z_]+:|$)", "i");
    const match = text.match(regex);
    if (!match) return [];
    return match[1]
      .split("\n")
      .map(l => l.replace(/^[-*•]\s*/, "").replace(/\*\*/g, "").trim())
      .filter(l => l.length > 2);
  }

  function extractField(text, label) {
    const match = text.match(new RegExp(label + ":\\s*(.+)", "i"));
    return match ? match[1].replace(/\*\*/g, "").trim() : "";
  }

  const crop    = extractField(raw, "CROP");
  const disease = extractField(raw, "DISEASE");
  const chemEn  = extractSection(raw, "CHEMICAL_EN");
  const chemMr  = extractSection(raw, "CHEMICAL_MR");
  const orgEn   = extractSection(raw, "ORGANIC_EN");
  const orgMr   = extractSection(raw, "ORGANIC_MR");

  const title = crop && disease ? `${crop} — ${disease}` : (crop || disease || "AI Vision Analysis");
  document.getElementById("disease-name").innerHTML = "<b>Diagnosis:</b> " + title;

  let html = "";

  if (chemEn.length) {
    html += "<h4>🧪 Chemical Treatment (English):</h4><ul>";
    chemEn.forEach(s => html += "<li>" + s + "</li>");
    html += "</ul>";
  }

  if (chemMr.length) {
    html += "<h4>🧪 Chemical Treatment (मराठी):</h4><ul>";
    chemMr.forEach(s => html += "<li>" + s + "</li>");
    html += "</ul>";
  }

  if (orgEn.length) {
    html += "<h4 style='color:green;'>🌿 Organic Treatment (English):</h4><ul>";
    orgEn.forEach(s => html += "<li>" + s + "</li>");
    html += "</ul>";
  }

  if (orgMr.length) {
    html += "<h4 style='color:green;'>🌿 सेंद्रिय उपाय (मराठी):</h4><ul>";
    orgMr.forEach(s => html += "<li>" + s + "</li>");
    html += "</ul>";
  }

  if (!html) {
    html = "<p>" + raw.replace(/\*\*/g, "").replace(/\n/g, "<br>") + "</p>";
  }

  document.getElementById("remedy-content").innerHTML = html;

  lastEnglishSteps = [...chemEn, ...orgEn];
  lastMarathiSteps = [...chemMr, ...orgMr];
  lastAIResponse = "";

  speakAdvice(lastEnglishSteps, lastMarathiSteps);
}

// ===============================
// GOOGLE BROWSER VOICE (OFFLINE)
// ===============================
function speakAdvice(steps_en, steps_mr) {

  const voices = speechSynthesis.getVoices();

  const marathiVoice = voices.find(v =>
    v.lang === "mr-IN"
  ) || voices.find(v =>
    v.lang.includes("mr") || v.lang.includes("hi")
  );

  const englishVoice = voices.find(v =>
    v.lang === "en-IN"
  ) || voices.find(v =>
    v.lang.includes("en")
  );

  const mrText = steps_mr.join(". ");
  const enText = steps_en.join(". ");

  const utterMr = new SpeechSynthesisUtterance(mrText);
  utterMr.lang = "mr-IN";
  if (marathiVoice) utterMr.voice = marathiVoice;

  const utterEn = new SpeechSynthesisUtterance(enText);
  utterEn.lang = "en-IN";
  if (englishVoice) utterEn.voice = englishVoice;

  speechSynthesis.cancel();
  speechSynthesis.speak(utterMr);
  speechSynthesis.speak(utterEn);
}

// ===============================
// GOOGLE BROWSER VOICE (ONLINE)
// ===============================
function speakAI(text) {

  const voices = speechSynthesis.getVoices();

  const englishVoice = voices.find(v =>
    v.lang === "en-IN"
  ) || voices.find(v =>
    v.lang.includes("en")
  );

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "en-IN";
  if (englishVoice) utter.voice = englishVoice;

  speechSynthesis.cancel();
  speechSynthesis.speak(utter);
}

// ===============================
// SPEAK BUTTON
// ===============================
function speakResult() {
  if (lastAIResponse) {
    speakAI(lastAIResponse);
  } else {
    speakAdvice(lastEnglishSteps, lastMarathiSteps);
  }
}

// ===============================
// INIT VOICE
// ===============================
function initVoice() {
  speechSynthesis.getVoices();
  alert("Google voice ready!");
}

// ===============================
// LOAD WEATHER (PUNE)
// ===============================
function loadWeather() {
  fetch("/weather")
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        document.getElementById("weather-info").innerHTML =
          "⚠ हवामान उपलब्ध नाही";
        return;
      }

      document.getElementById("weather-info").innerHTML =
        `🌤 आजचे हवामान: ${data.temp}°C | ${data.advice}`;
    })
    .catch(() => {
      document.getElementById("weather-info").innerHTML =
        "⚠ हवामान उपलब्ध नाही";
    });
}

window.addEventListener("load", loadWeather);