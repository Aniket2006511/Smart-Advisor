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

      // If error in online → fallback to offline
      if (data.error && selectedMode === "online") {
        console.log("Online failed. Switching to offline...");
        fallbackToOffline(formData);
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
        fallbackToOffline(formData);
      } else {
        alert("Prediction failed.");
      }
    });
}

// ===============================
// FALLBACK TO OFFLINE
// ===============================
function fallbackToOffline(formData) {

  fetch("/predict", {
    method: "POST",
    body: formData,
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

  let stepsHTML = "<h4>English:</h4><ul>";
  data.steps_en.forEach(step => {
    stepsHTML += "<li>" + step + "</li>";
  });
  stepsHTML += "</ul>";

  stepsHTML += "<h4>मराठी:</h4><ul>";
  data.steps_mr.forEach(step => {
    stepsHTML += "<li>" + step + "</li>";
  });
  stepsHTML += "</ul>";

  document.getElementById("remedy-content").innerHTML = stepsHTML;

  lastEnglishSteps = data.steps_en;
  lastMarathiSteps = data.steps_mr;
  lastAIResponse = "";

  speakAdvice(lastEnglishSteps, lastMarathiSteps);
}

// ===============================
// ONLINE RESULT (Gemini)
// ===============================
function showOnlineResult(data) {

  document.getElementById("result-section").style.display = "block";

  document.getElementById("disease-name").innerHTML =
    "<b>Diagnosis:</b> " + data.diagnosis;

  document.getElementById("remedy-content").innerHTML =
    "<div>" + data.ai_response.replace(/\n/g, "<br>") + "</div>";

  lastAIResponse = data.ai_response;
  lastEnglishSteps = [];
  lastMarathiSteps = [];

  speakAI(data.ai_response);
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
