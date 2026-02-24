import os
import json
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
YOUR_API_KEY = None
try:
    YOUR_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
except Exception:
    pass

if not YOUR_API_KEY:
    YOUR_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not YOUR_API_KEY:
    st.error("Missing ASSEMBLYAI_API_KEY. Add it to Streamlit secrets or a .env file.")
    st.stop()

ASSEMBLYAI_SAMPLE_RATE = 16000
TOKEN_URL = "https://streaming.assemblyai.com/v3/token"
WS_BASE = "wss://streaming.assemblyai.com/v3/ws"

# --- OpenAI Configuration ---
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets or a .env file.")
    st.stop()

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# --- Medical Keyterms by Specialty ---
MEDICAL_KEYTERMS = {
    "General Practice": [
        "hypertension", "diabetes mellitus", "hyperlipidemia",
        "metformin", "lisinopril", "atorvastatin", "amlodipine",
        "hemoglobin A1c", "blood pressure", "BMI",
        "chief complaint", "review of systems", "auscultation",
        "palpation", "percussion", "vital signs", "ibuprofen",
        "acetaminophen", "amoxicillin", "prednisone"
    ],
    "Cardiology": [
        "ejection fraction", "coronary artery disease", "ST elevation",
        "troponin", "echocardiogram", "electrocardiogram", "ECG",
        "atrial fibrillation", "heart failure", "stent",
        "angioplasty", "beta blocker", "metoprolol", "warfarin",
        "anticoagulation", "chest pain", "dyspnea", "palpitations",
        "myocardial infarction", "cardiac catheterization"
    ],
    "Endocrinology": [
        "hemoglobin A1c", "insulin resistance", "thyroid",
        "levothyroxine", "TSH", "T3", "T4", "glucose tolerance",
        "diabetic neuropathy", "retinopathy", "metformin",
        "insulin glargine", "GLP-1 agonist", "semaglutide",
        "Hashimoto's thyroiditis", "Graves' disease", "adrenal insufficiency"
    ],
    "Orthopedics": [
        "anterior cruciate ligament", "ACL", "meniscus",
        "arthroscopy", "MRI", "cortisone injection",
        "ibuprofen", "range of motion", "physical therapy",
        "fracture", "dislocation", "sprain", "rotator cuff",
        "carpal tunnel", "osteoarthritis", "bone density"
    ],
    "Psychiatry": [
        "sertraline", "fluoxetine", "cognitive behavioral therapy",
        "major depressive disorder", "generalized anxiety",
        "SSRI", "SNRI", "benzodiazepine", "PHQ-9", "GAD-7",
        "bipolar disorder", "schizophrenia", "PTSD",
        "insomnia", "panic disorder", "suicidal ideation"
    ],
}


# --- API Helper Functions ---

def get_temporary_token():
    """Generate a one-time temporary token for browser-side AssemblyAI auth."""
    resp = requests.get(
        TOKEN_URL,
        params={"expires_in_seconds": 480},
        headers={"Authorization": YOUR_API_KEY},
    )
    resp.raise_for_status()
    return resp.json()["token"]


def call_llm(system_prompt, user_content, max_tokens=4000, temperature=0.1):
    """Call OpenAI API for post-processing."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4.1-nano-2025-04-14",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(OPENAI_API_URL, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_soap_note(transcript, specialty):
    """Generate a SOAP note from the encounter transcript via LLM Gateway."""
    system_prompt = f"""You are an expert medical scribe specializing in {specialty}. \
Generate a structured SOAP note from this medical encounter transcript.

Format your response with these exact section headers:
## Subjective
Patient's chief complaint, history of present illness, review of systems, \
and relevant past medical/surgical/family/social history as reported by the patient.

## Objective
Provider observations, physical examination findings, vital signs, \
and diagnostic test results mentioned during the encounter.

## Assessment
Clinical impressions, differential diagnoses, and diagnostic reasoning.

## Plan
Treatment plan, medications prescribed (with dosages), \
follow-up instructions, referrals, and patient education provided.

Use appropriate medical terminology. Only include information explicitly \
stated in the transcript. Do not fabricate clinical data."""

    return call_llm(system_prompt, f"Encounter Transcript:\n\n{transcript}")


def redact_pii(transcript):
    """Redact PII from the transcript via LLM Gateway."""
    system_prompt = """You are a HIPAA compliance specialist. Analyze the following medical \
encounter transcript and redact all personally identifiable information (PII).

Replace each PII instance with the appropriate label in brackets:
- Person names -> [PERSON_NAME]
- Dates of birth -> [DATE_OF_BIRTH]
- Phone numbers -> [PHONE_NUMBER]
- Email addresses -> [EMAIL_ADDRESS]
- Social security numbers -> [SSN]
- Medical record numbers -> [MRN]
- Addresses/locations -> [ADDRESS]
- Organizations/employers -> [ORGANIZATION]
- Insurance IDs -> [INSURANCE_ID]

Maintain ALL medical terminology, diagnoses, medications, and clinical details unchanged.
Only redact information that could identify a specific individual.
Return ONLY the redacted transcript, maintaining the exact same format with speaker labels."""

    return call_llm(system_prompt, f"Transcript:\n\n{transcript}")


def analyze_sentiment(transcript):
    """Analyze sentiment per speaker turn via LLM Gateway."""
    system_prompt = """You are a clinical communication analyst. Analyze the sentiment \
of each speaker turn in this medical encounter transcript.

For each turn, assess the emotional tone. Then provide an overall summary.

Return your analysis as valid JSON with this exact structure:
{
  "turns": [
    {
      "speaker": "Doctor" or "Patient",
      "excerpt": "first 8-10 words of the turn...",
      "sentiment": "POSITIVE" or "NEUTRAL" or "NEGATIVE",
      "confidence": "HIGH" or "MEDIUM" or "LOW",
      "reason": "one sentence explanation"
    }
  ],
  "patient_summary": "2-3 sentence summary of patient's overall emotional state",
  "overall_patient_sentiment": "POSITIVE" or "NEUTRAL" or "NEGATIVE",
  "overall_doctor_sentiment": "POSITIVE" or "NEUTRAL" or "NEGATIVE"
}

Return ONLY valid JSON, no markdown code fences or other text."""

    return call_llm(system_prompt, f"Transcript:\n\n{transcript}")


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Medical Scribe | AssemblyAI",
    page_icon="🏥",
    layout="wide",
)

# --- Session State Initialization ---
defaults = {
    "encounter_active": False,
    "encounter_transcript": "",
    "encounter_turns": [],
    "soap_note": None,
    "redacted_transcript": None,
    "sentiment_results": None,
    "selected_specialty": "General Practice",
    "processing": False,
    "processing_step": "",
    "custom_keyterms": "",
    "patient_context": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Generate a fresh token for the browser component
token = get_temporary_token()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Encounter Setup")

    specialty = st.selectbox(
        "Medical Specialty",
        list(MEDICAL_KEYTERMS.keys()),
        index=list(MEDICAL_KEYTERMS.keys()).index(st.session_state.selected_specialty),
        key="specialty_select",
    )
    st.session_state.selected_specialty = specialty

    st.text_area(
        "Patient Context (optional)",
        placeholder="e.g. 65yo male, history of Type 2 diabetes, on metformin 1000mg BID",
        key="patient_context_input",
        height=80,
    )

    st.text_input(
        "Additional Keyterms (comma-separated)",
        placeholder="e.g. ozempic, tirzepatide, GFR",
        key="custom_keyterms_input",
    )

    # Build the active keyterms list
    active_keyterms = MEDICAL_KEYTERMS[specialty].copy()
    custom = st.session_state.get("custom_keyterms_input", "")
    if custom.strip():
        active_keyterms.extend([t.strip() for t in custom.split(",") if t.strip()])

    with st.expander("Active Keyterms", expanded=False):
        st.caption(", ".join(active_keyterms))

    st.markdown("---")
    st.markdown("### AssemblyAI Features")

    feature_cols = st.columns(2)
    features = [
        ("Streaming STT", True),
        ("Medical Keyterms", True),
        ("Speaker Diarization", True),
        ("PII Redaction", True),
        ("SOAP Notes", True),
        ("Sentiment Analysis", True),
    ]
    for i, (name, active) in enumerate(features):
        col = feature_cols[i % 2]
        if active:
            col.markdown(
                f'<span style="background:#dcfce7;color:#166534;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;font-weight:600;">'
                f'{name}</span>',
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<span style="background:#f1f5f9;color:#94a3b8;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;font-weight:600;">'
                f'{name}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
1. Select a **specialty** and add optional context
2. Click **Start Encounter** to begin live transcription
3. Speak naturally -- turns are transcribed in real-time
4. Click **End Encounter** to generate:
   - SOAP Notes via LLM Gateway
   - PII-redacted transcript
   - Sentiment analysis
""")


# --- Main Area ---
st.markdown(
    '<h1 style="margin-bottom:0;">AI Medical Scribe</h1>',
    unsafe_allow_html=True,
)
st.caption("Real-time clinical documentation powered by AssemblyAI Streaming STT + OpenAI")

# Feature badges strip
st.markdown(
    """<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;">
    <span style="background:#eff6ff;color:#2563eb;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">Streaming STT</span>
    <span style="background:#f0fdf4;color:#16a34a;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">Medical Keyterms</span>
    <span style="background:#fdf4ff;color:#a21caf;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">Speaker Diarization</span>
    <span style="background:#fef3c7;color:#b45309;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">PII Redaction</span>
    <span style="background:#fce7f3;color:#be185d;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">SOAP Notes</span>
    <span style="background:#e0e7ff;color:#4338ca;padding:3px 10px;border-radius:14px;
    font-size:11px;font-weight:600;">Sentiment Analysis</span>
    </div>""",
    unsafe_allow_html=True,
)

# Tabs
tab_transcript, tab_soap, tab_pii, tab_analysis = st.tabs([
    "Live Transcript",
    "SOAP Notes",
    "PII Redaction",
    "Sentiment Analysis",
])

# --- Tab 1: Live Transcript (HTML/JS Audio Component) ---
with tab_transcript:
    # Build keyterms JSON for JavaScript injection
    keyterms_json = json.dumps(active_keyterms)
    patient_ctx = st.session_state.get("patient_context_input", "")

    AUDIO_COMPONENT_HTML = f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}

  .controls {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .btn {{
    padding: 10px 24px;
    font-size: 15px;
    font-weight: 600;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .btn:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}
  .btn-start {{ background: #0891b2; }}
  .btn-start:hover:not(:disabled) {{ background: #0e7490; }}
  .btn-end {{ background: #dc2626; }}
  .btn-end:hover:not(:disabled) {{ background: #b91c1c; }}
  .btn-new {{ background: #6366f1; }}
  .btn-new:hover:not(:disabled) {{ background: #4f46e5; }}
  .btn-process {{ background: #7c3aed; }}
  .btn-process:hover:not(:disabled) {{ background: #6d28d9; }}

  #status {{
    font-size: 13px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
  }}

  .transcript-container {{
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    min-height: 300px;
    max-height: 500px;
    overflow-y: auto;
    padding: 16px;
    background: #ffffff;
  }}

  .turn {{
    padding: 8px 14px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 14px;
    line-height: 1.6;
  }}
  .turn-doctor {{
    background: #eff6ff;
    border-left: 4px solid #2563eb;
  }}
  .turn-patient {{
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
  }}
  .turn-partial {{
    background: #f8fafc;
    border-left: 4px solid #94a3b8;
    opacity: 0.7;
    font-style: italic;
  }}
  .speaker-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }}
  .speaker-doctor {{ color: #2563eb; }}
  .speaker-patient {{ color: #16a34a; }}
  .speaker-partial {{ color: #94a3b8; }}

  .waiting-msg {{
    color: #94a3b8;
    font-style: italic;
    text-align: center;
    padding: 40px 0;
  }}

  .encounter-info {{
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
    font-size: 12px;
    color: #64748b;
  }}
  .encounter-info span {{ font-weight: 600; color: #334155; }}
</style>

<div class="controls">
  <button id="startBtn" class="btn btn-start" onclick="startRecording()">
    Start Encounter
  </button>
  <button id="stopBtn" class="btn btn-end" onclick="stopRecording()" disabled>
    End Encounter
  </button>
  <button id="newBtn" class="btn btn-new" onclick="newEncounter()" style="display:none;">
    New Encounter
  </button>
  <div id="status" style="background:#f1f5f9;color:#64748b;">Ready</div>
</div>

<div class="encounter-info">
  <div>Duration: <span id="duration">00:00</span></div>
  <div>Turns: <span id="turnCount">0</span></div>
  <div>Words: <span id="wordCount">0</span></div>
</div>

<div id="transcript" class="transcript-container">
  <div class="waiting-msg">Click "Start Encounter" to begin real-time transcription</div>
</div>

<textarea id="hiddenTranscript" style="width:100%;height:120px;margin-top:12px;
  font-family:monospace;font-size:12px;padding:10px;border:1px solid #e2e8f0;
  border-radius:8px;background:#f8fafc;color:#334155;display:none;"
  placeholder="Transcript will appear here after the encounter ends..."></textarea>

<script>
const SAMPLE_RATE = {ASSEMBLYAI_SAMPLE_RATE};
const TOKEN = "{token}";
const KEYTERMS = {keyterms_json};

// Build WebSocket URL with medical parameters
const wsParams = new URLSearchParams({{
  sample_rate: SAMPLE_RATE,
  format_turns: "true",
  encoding: "pcm_s16le",
  token: TOKEN,
  end_of_turn_confidence_threshold: "0.7",
  min_end_of_turn_silence_when_confident: "800",
  max_turn_silence: "3600",
}});
// Add keyterms if present
if (KEYTERMS && KEYTERMS.length > 0) {{
  wsParams.set("keyterms_prompt", JSON.stringify(KEYTERMS));
}}
const WS_URL = "{WS_BASE}?" + wsParams.toString();

let audioContext = null;
let mediaStream = null;
let ws = null;
let transcripts = [];      // Array of {{ text, speaker }} objects
let currentPartial = "";
let startTime = null;
let timerInterval = null;
let totalWords = 0;

function setStatus(msg, bg, fg) {{
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.background = bg || "#f1f5f9";
  el.style.color = fg || "#64748b";
}}

function updateStats() {{
  document.getElementById("turnCount").textContent = transcripts.length;
  document.getElementById("wordCount").textContent = totalWords;
}}

function startTimer() {{
  startTime = Date.now();
  timerInterval = setInterval(() => {{
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    document.getElementById("duration").textContent = mins + ":" + secs;
  }}, 1000);
}}

function stopTimer() {{
  if (timerInterval) clearInterval(timerInterval);
}}

function getSpeaker(index) {{
  // Alternate turns: first speaker = Doctor (typically asks initial question)
  return index % 2 === 0 ? "Doctor" : "Patient";
}}

function renderTranscript() {{
  const el = document.getElementById("transcript");
  if (transcripts.length === 0 && !currentPartial) {{
    el.innerHTML = '<div class="waiting-msg">Listening... speak to begin the encounter</div>';
    return;
  }}

  let html = "";
  transcripts.forEach((t, i) => {{
    const speaker = t.speaker;
    const cls = speaker === "Doctor" ? "doctor" : "patient";
    html += '<div class="turn turn-' + cls + '">';
    html += '<div class="speaker-label speaker-' + cls + '">' + speaker + '</div>';
    html += '<div>' + t.text + '</div>';
    html += '</div>';
  }});

  if (currentPartial) {{
    const nextSpeaker = getSpeaker(transcripts.length);
    html += '<div class="turn turn-partial">';
    html += '<div class="speaker-label speaker-partial">' + nextSpeaker + ' (speaking...)</div>';
    html += '<div>' + currentPartial + '</div>';
    html += '</div>';
  }}

  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
}}

function buildTranscriptText() {{
  return transcripts.map(t => t.speaker + ": " + t.text).join("\\n\\n");
}}

async function startRecording() {{
  try {{
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    document.getElementById("newBtn").style.display = "none";
    transcripts = [];
    currentPartial = "";
    totalWords = 0;
    renderTranscript();
    setStatus("Connecting...", "#fef3c7", "#b45309");

    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {{
      setStatus("Recording", "#dcfce7", "#166534");
      startTimer();
    }};

    ws.onmessage = (event) => {{
      const data = JSON.parse(event.data);
      if (data.type === "Turn") {{
        const text = data.transcript || "";
        if (data.turn_is_formatted && text.trim()) {{
          const speaker = getSpeaker(transcripts.length);
          transcripts.push({{ text: text.trim(), speaker }});
          totalWords += text.trim().split(/\\s+/).length;
          currentPartial = "";
        }} else if (text.trim()) {{
          currentPartial = text.trim();
        }}
        renderTranscript();
        updateStats();
      }} else if (data.type === "Termination") {{
        setStatus("Session ended", "#f1f5f9", "#64748b");
      }} else if (data.type === "Begin") {{
        console.log("[WS] Session began:", data.id);
      }}
    }};

    ws.onerror = (err) => {{
      console.error("[WS] Error:", err);
      setStatus("WebSocket error", "#fef2f2", "#dc2626");
    }};

    ws.onclose = () => {{
      console.log("[WS] Closed");
    }};

    mediaStream = await navigator.mediaDevices.getUserMedia({{
      audio: {{
        sampleRate: SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }},
      video: false,
    }});

    audioContext = new AudioContext({{ sampleRate: SAMPLE_RATE }});
    const source = audioContext.createMediaStreamSource(mediaStream);

    const bufferSize = 4096;
    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (e) => {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {{
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }}
      ws.send(int16.buffer);
    }};

    source.connect(processor);
    processor.connect(audioContext.destination);

    window._sttProcessor = processor;
    window._sttSource = source;

  }} catch (err) {{
    console.error("Start error:", err);
    setStatus("Error: " + err.message, "#fef2f2", "#dc2626");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
  }}
}}

function stopRecording() {{
  document.getElementById("stopBtn").disabled = true;
  setStatus("Stopping...", "#fef3c7", "#b45309");
  stopTimer();

  if (window._sttProcessor) {{
    window._sttProcessor.disconnect();
    window._sttProcessor = null;
  }}
  if (window._sttSource) {{
    window._sttSource.disconnect();
    window._sttSource = null;
  }}
  if (audioContext) {{
    audioContext.close();
    audioContext = null;
  }}
  if (mediaStream) {{
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }}

  if (ws && ws.readyState === WebSocket.OPEN) {{
    ws.send(JSON.stringify({{ type: "Terminate" }}));
    setTimeout(() => {{
      ws.close();
      ws = null;
      onEncounterStopped();
    }}, 1500);
  }} else {{
    onEncounterStopped();
  }}
}}

function onEncounterStopped() {{
  if (transcripts.length > 0) {{
    document.getElementById("newBtn").style.display = "inline-block";
    onEncounterComplete();
  }} else {{
    setStatus("No speech detected", "#f1f5f9", "#64748b");
    document.getElementById("newBtn").style.display = "inline-block";
  }}
}}

function onEncounterComplete() {{
  // Show transcript in the copyable textarea
  const transcriptText = buildTranscriptText();
  const hiddenEl = document.getElementById("hiddenTranscript");
  if (hiddenEl) {{
    hiddenEl.value = transcriptText;
    hiddenEl.style.display = "block";
  }}

  setStatus("Encounter complete — copy transcript below, paste into the text area, and click Generate Notes", "#eff6ff", "#2563eb");
}}

function newEncounter() {{
  transcripts = [];
  currentPartial = "";
  totalWords = 0;
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
  document.getElementById("newBtn").style.display = "none";
  document.getElementById("duration").textContent = "00:00";
  updateStats();
  setStatus("Ready", "#f1f5f9", "#64748b");
  renderTranscript();
  const hiddenEl = document.getElementById("hiddenTranscript");
  if (hiddenEl) hiddenEl.value = "";
}}
</script>
"""

    components.html(AUDIO_COMPONENT_HTML, height=650)

    # Streamlit-native transcript input and processing trigger
    st.markdown("---")
    st.markdown("**After the encounter ends**, paste the transcript from the box above:")

    transcript_input = st.text_area(
        "Encounter Transcript",
        value=st.session_state.encounter_transcript,
        height=150,
        placeholder="Paste the encounter transcript here after clicking End Encounter...",
        key="transcript_input",
    )

    col_gen, col_clear = st.columns([1, 1])
    with col_gen:
        generate_clicked = st.button(
            "Generate SOAP Notes, PII Redaction & Sentiment",
            type="primary",
            disabled=st.session_state.processing or not transcript_input.strip(),
            use_container_width=True,
        )
    with col_clear:
        clear_clicked = st.button(
            "Clear / New Encounter",
            use_container_width=True,
        )

    if generate_clicked and transcript_input.strip():
        st.session_state.encounter_transcript = transcript_input.strip()
        st.session_state.processing = True
        st.session_state.soap_note = None
        st.session_state.redacted_transcript = None
        st.session_state.sentiment_results = None
        st.rerun()

    if clear_clicked:
        st.session_state.encounter_transcript = ""
        st.session_state.soap_note = None
        st.session_state.redacted_transcript = None
        st.session_state.sentiment_results = None
        st.session_state.processing = False
        st.rerun()


# --- Post-Encounter Processing ---
if st.session_state.processing and st.session_state.encounter_transcript:
    transcript = st.session_state.encounter_transcript
    specialty = st.session_state.selected_specialty

    with st.spinner("Generating SOAP Notes via LLM Gateway..."):
        try:
            st.session_state.soap_note = generate_soap_note(transcript, specialty)
        except Exception as e:
            st.session_state.soap_note = f"Error generating SOAP note: {e}"

    with st.spinner("Redacting PII via LLM Gateway..."):
        try:
            st.session_state.redacted_transcript = redact_pii(transcript)
        except Exception as e:
            st.session_state.redacted_transcript = f"Error redacting PII: {e}"

    with st.spinner("Analyzing Sentiment via LLM Gateway..."):
        try:
            st.session_state.sentiment_results = analyze_sentiment(transcript)
        except Exception as e:
            st.session_state.sentiment_results = json.dumps({
                "error": str(e),
                "turns": [],
                "patient_summary": "Error analyzing sentiment",
                "overall_patient_sentiment": "UNKNOWN",
                "overall_doctor_sentiment": "UNKNOWN",
            })

    st.session_state.processing = False
    st.rerun()


# --- Tab 2: SOAP Notes ---
with tab_soap:
    if st.session_state.soap_note:
        st.markdown("### Generated SOAP Note")
        st.caption(f"Specialty: {st.session_state.selected_specialty} | Generated via LLM Gateway")
        st.markdown("---")
        st.markdown(st.session_state.soap_note)
    elif st.session_state.processing:
        st.info("Generating SOAP notes... please wait.")
    else:
        st.info("Complete an encounter to generate SOAP notes. Start recording, speak, then click 'End Encounter' followed by 'Generate Notes'.")


# --- Tab 3: PII Redaction ---
with tab_pii:
    if st.session_state.redacted_transcript and st.session_state.encounter_transcript:
        st.markdown("### PII Redaction")
        st.caption("Personally Identifiable Information automatically detected and redacted")

        pii_policies = [
            "Person Names", "Dates of Birth", "Phone Numbers",
            "Email Addresses", "SSN", "Medical Record Numbers",
            "Addresses", "Organizations", "Insurance IDs",
        ]
        st.markdown(
            " ".join(
                f'<span style="background:#fef3c7;color:#b45309;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;font-weight:600;margin:2px;">'
                f'{p}</span>'
                for p in pii_policies
            ),
            unsafe_allow_html=True,
        )
        st.markdown("")

        show_original = st.toggle("Show Original (unredacted)", value=False)

        if show_original:
            st.warning("Displaying original transcript with PII visible")
            st.text_area(
                "Original Transcript",
                st.session_state.encounter_transcript,
                height=300,
                disabled=True,
            )
        else:
            st.success("PII has been redacted from the transcript")
            st.text_area(
                "Redacted Transcript",
                st.session_state.redacted_transcript,
                height=300,
                disabled=True,
            )
    elif st.session_state.processing:
        st.info("Redacting PII... please wait.")
    else:
        st.info("Complete an encounter to see PII redaction results.")


# --- Tab 4: Sentiment Analysis ---
with tab_analysis:
    if st.session_state.sentiment_results:
        st.markdown("### Sentiment Analysis")
        st.caption("Emotional tone analysis per speaker turn via LLM Gateway")

        try:
            raw = st.session_state.sentiment_results
            if isinstance(raw, str):
                # Strip markdown code fences if present
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                sentiment_data = json.loads(cleaned.strip())
            else:
                sentiment_data = raw

            # Overall sentiment cards
            col1, col2 = st.columns(2)

            sentiment_colors = {
                "POSITIVE": ("#dcfce7", "#166534"),
                "NEUTRAL": ("#f1f5f9", "#475569"),
                "NEGATIVE": ("#fef2f2", "#dc2626"),
            }

            patient_sent = sentiment_data.get("overall_patient_sentiment", "NEUTRAL")
            doctor_sent = sentiment_data.get("overall_doctor_sentiment", "NEUTRAL")

            p_bg, p_fg = sentiment_colors.get(patient_sent, ("#f1f5f9", "#475569"))
            d_bg, d_fg = sentiment_colors.get(doctor_sent, ("#f1f5f9", "#475569"))

            col1.markdown(
                f'<div style="background:{p_bg};color:{p_fg};padding:16px;border-radius:10px;'
                f'text-align:center;">'
                f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;">Patient Sentiment</div>'
                f'<div style="font-size:24px;font-weight:700;margin:4px 0;">{patient_sent}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            col2.markdown(
                f'<div style="background:{d_bg};color:{d_fg};padding:16px;border-radius:10px;'
                f'text-align:center;">'
                f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;">Doctor Sentiment</div>'
                f'<div style="font-size:24px;font-weight:700;margin:4px 0;">{doctor_sent}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Patient summary
            summary = sentiment_data.get("patient_summary", "")
            if summary:
                st.markdown(f"**Patient Summary:** {summary}")

            st.markdown("---")

            # Per-turn breakdown
            turns = sentiment_data.get("turns", [])
            if turns:
                st.markdown("#### Turn-by-Turn Analysis")
                for turn in turns:
                    sent = turn.get("sentiment", "NEUTRAL")
                    bg, fg = sentiment_colors.get(sent, ("#f1f5f9", "#475569"))
                    speaker = turn.get("speaker", "Unknown")
                    excerpt = turn.get("excerpt", "")
                    confidence = turn.get("confidence", "")
                    reason = turn.get("reason", "")

                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
                        f'border-bottom:1px solid #e2e8f0;">'
                        f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;'
                        f'font-size:11px;font-weight:700;min-width:70px;text-align:center;">{sent}</span>'
                        f'<span style="font-weight:600;min-width:60px;color:'
                        f'{"#2563eb" if speaker == "Doctor" else "#16a34a"}">{speaker}</span>'
                        f'<span style="color:#64748b;font-size:13px;">"{excerpt}..." '
                        f'<span style="color:#94a3b8;">({confidence})</span></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if reason:
                        st.caption(f"  {reason}")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.warning("Could not parse sentiment analysis results as structured data.")
            st.text_area("Raw Sentiment Analysis", st.session_state.sentiment_results, height=300, disabled=True)

    elif st.session_state.processing:
        st.info("Analyzing sentiment... please wait.")
    else:
        st.info("Complete an encounter to see sentiment analysis.")
