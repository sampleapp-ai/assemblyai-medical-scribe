import os
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


def get_temporary_token():
    """Generate a one-time temporary token for browser-side AssemblyAI auth."""
    resp = requests.get(
        TOKEN_URL,
        params={"expires_in_seconds": 480},
        headers={"Authorization": YOUR_API_KEY},
    )
    resp.raise_for_status()
    return resp.json()["token"]


# --- Streamlit UI ---
st.set_page_config(page_title="AssemblyAI Streaming STT", layout="wide")
st.title("AssemblyAI Streaming STT")
st.caption("Real-time speech-to-text powered by AssemblyAI")

# Session state
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

# Generate a fresh token for the browser component
token = get_temporary_token()

# Status / transcript placeholders
status_placeholder = st.empty()
status_placeholder.info("Ready. Click **Start Recording** to begin.")

st.subheader("Live Transcript")
transcript_placeholder = st.empty()

# Display any previously collected transcripts
if st.session_state.transcripts:
    transcript_placeholder.markdown("\n\n".join(st.session_state.transcripts))

# --- Browser-side audio capture + AssemblyAI streaming component ---
# Everything below runs in an iframe: mic capture, WebSocket to AssemblyAI,
# and transcript display. No WebRTC / TURN server needed.

AUDIO_COMPONENT_HTML = f"""
<div id="controls" style="margin-bottom: 1rem;">
  <button id="startBtn" onclick="startRecording()"
    style="padding: 10px 24px; font-size: 16px; background: #4CAF50; color: white;
           border: none; border-radius: 6px; cursor: pointer; margin-right: 8px;">
    Start Recording
  </button>
  <button id="stopBtn" onclick="stopRecording()" disabled
    style="padding: 10px 24px; font-size: 16px; background: #f44336; color: white;
           border: none; border-radius: 6px; cursor: pointer; opacity: 0.5;">
    Stop Recording
  </button>
  <span id="status" style="margin-left: 12px; font-size: 14px; color: #666;"></span>
</div>
<div id="transcript"
  style="white-space: pre-wrap; font-family: sans-serif; font-size: 15px;
         line-height: 1.6; padding: 12px; border: 1px solid #ddd; border-radius: 8px;
         min-height: 120px; max-height: 400px; overflow-y: auto; background: #fafafa;">
</div>

<script>
const SAMPLE_RATE = {ASSEMBLYAI_SAMPLE_RATE};
const TOKEN = "{token}";
const WS_URL = "{WS_BASE}?sample_rate=" + SAMPLE_RATE + "&format_turns=true&encoding=pcm_s16le&token=" + TOKEN;

let audioContext = null;
let mediaStream = null;
let workletNode = null;
let ws = null;
let transcripts = [];
let currentPartial = "";

function setStatus(msg, color) {{
  document.getElementById("status").textContent = msg;
  document.getElementById("status").style.color = color || "#666";
}}

function renderTranscript() {{
  const el = document.getElementById("transcript");
  let text = transcripts.join("\\n\\n");
  if (currentPartial) {{
    if (text) text += "\\n\\n";
    text += currentPartial + " ...";
  }}
  el.textContent = text || "(waiting for speech...)";
  el.scrollTop = el.scrollHeight;
}}

async function startRecording() {{
  try {{
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    document.getElementById("stopBtn").style.opacity = "1";
    document.getElementById("startBtn").style.opacity = "0.5";
    transcripts = [];
    currentPartial = "";
    renderTranscript();
    setStatus("Connecting...", "#FF9800");

    // 1. Open WebSocket to AssemblyAI
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {{
      setStatus("Recording... speak now", "#4CAF50");
      console.log("[WS] Connected to AssemblyAI");
    }};

    ws.onmessage = (event) => {{
      const data = JSON.parse(event.data);
      if (data.type === "Turn") {{
        const text = data.transcript || "";
        if (data.turn_is_formatted && text.trim()) {{
          // Final formatted turn — add to list
          transcripts.push(text);
          currentPartial = "";
        }} else if (text.trim()) {{
          // Partial / unformatted — show as in-progress
          currentPartial = text;
        }}
        renderTranscript();
      }} else if (data.type === "Termination") {{
        console.log("[WS] Session terminated");
        setStatus("Session ended", "#666");
      }} else if (data.type === "Begin") {{
        console.log("[WS] Session began:", data.id);
      }}
    }};

    ws.onerror = (err) => {{
      console.error("[WS] Error:", err);
      setStatus("WebSocket error", "#f44336");
    }};

    ws.onclose = () => {{
      console.log("[WS] Closed");
    }};

    // 2. Capture microphone audio
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

    // Use ScriptProcessorNode (widely supported) to get raw PCM
    const bufferSize = 4096;
    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (e) => {{
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const float32 = e.inputBuffer.getChannelData(0);
      // Convert float32 [-1,1] to int16
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {{
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }}
      ws.send(int16.buffer);
    }};

    source.connect(processor);
    processor.connect(audioContext.destination);

    // Store processor ref for cleanup
    window._sttProcessor = processor;
    window._sttSource = source;

  }} catch (err) {{
    console.error("Start error:", err);
    setStatus("Error: " + err.message, "#f44336");
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    document.getElementById("startBtn").style.opacity = "1";
    document.getElementById("stopBtn").style.opacity = "0.5";
  }}
}}

function stopRecording() {{
  document.getElementById("stopBtn").disabled = true;
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").style.opacity = "0.5";
  setStatus("Stopping...", "#FF9800");

  // Disconnect audio
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

  // Send Terminate to AssemblyAI
  if (ws && ws.readyState === WebSocket.OPEN) {{
    ws.send(JSON.stringify({{ type: "Terminate" }}));
    setTimeout(() => {{
      ws.close();
      ws = null;
      setStatus("Stopped. Refresh the page to record again.", "#666");
    }}, 1500);
  }} else {{
    setStatus("Stopped. Refresh the page to record again.", "#666");
  }}
}}
</script>
"""

components.html(AUDIO_COMPONENT_HTML, height=500)
