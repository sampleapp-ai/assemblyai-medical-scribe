import os
import json
import threading
import queue
import time
import sys
import struct

import numpy as np
from scipy.signal import resample as scipy_resample
# import requests
import websocket
import av

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
YOUR_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

ASSEMBLYAI_SAMPLE_RATE = 16000
CONNECTION_PARAMS = {
    "sample_rate": ASSEMBLYAI_SAMPLE_RATE,
    "format_turns": "true",
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"
# LLM_GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"

# Target chunk size: 50ms at 16kHz mono = 800 samples = 1600 bytes (matches console app)
TARGET_CHUNK_SAMPLES = 800

# --- Shared state (not in st.session_state because threads need direct access) ---
# These are module-level singletons shared between the audio callback thread and main thread
audio_queue = queue.Queue(maxsize=500)
transcript_list = []
transcript_lock = threading.Lock()
stop_event = threading.Event()
ws_connection = None
ws_sender_thread = None
ws_receiver_thread = None

# Accumulator for resampled audio to build proper-sized chunks
_pcm_accumulator = bytearray()
_pcm_lock = threading.Lock()


# --- Audio Processing Callback ---
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Called by streamlit-webrtc for each audio frame.
    Converts to 16kHz mono PCM16 and enqueues for WebSocket sender.
    """
    global _pcm_accumulator

    try:
        # Get raw int16 samples from frame
        arr = frame.to_ndarray()  # shape: (channels, samples) for s16 planar
        source_rate = frame.sample_rate
        num_channels = len(frame.layout.channels)

        # Convert to float64 for resampling
        samples = arr.flatten().astype(np.float64)

        # Convert to mono
        if num_channels > 1:
            num_samples = len(samples) // num_channels
            samples = samples[:num_samples * num_channels].reshape(num_samples, num_channels).mean(axis=1)

        # Resample to 16kHz
        if source_rate != ASSEMBLYAI_SAMPLE_RATE:
            num_target = int(len(samples) * ASSEMBLYAI_SAMPLE_RATE / source_rate)
            if num_target > 0:
                samples = scipy_resample(samples, num_target)

        # Convert to int16 bytes
        pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

        # Accumulate and send in proper-sized chunks (800 samples = 1600 bytes)
        chunk_bytes = TARGET_CHUNK_SAMPLES * 2  # 2 bytes per int16 sample
        with _pcm_lock:
            _pcm_accumulator.extend(pcm)
            while len(_pcm_accumulator) >= chunk_bytes:
                chunk = bytes(_pcm_accumulator[:chunk_bytes])
                _pcm_accumulator = _pcm_accumulator[chunk_bytes:]
                try:
                    audio_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    audio_queue.put_nowait(chunk)
    except Exception as e:
        print(f"[AUDIO-CB] Error: {e}", file=sys.stderr)

    return frame


# --- WebSocket Management ---
def ws_sender_loop():
    """Read PCM bytes from audio_queue and send as binary frames to AssemblyAI."""
    global ws_connection
    chunks_sent = 0
    while not stop_event.is_set():
        try:
            pcm_bytes = audio_queue.get(timeout=0.1)
            if pcm_bytes and ws_connection:
                ws_connection.send_binary(pcm_bytes)
                chunks_sent += 1
                if chunks_sent % 100 == 0:
                    print(f"[WS-SENDER] Sent {chunks_sent} chunks ({len(pcm_bytes)} bytes each)", file=sys.stderr)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[WS-SENDER] Error: {e}", file=sys.stderr)
            break
    print(f"[WS-SENDER] Stopped. Total chunks sent: {chunks_sent}", file=sys.stderr)


def ws_receiver_loop():
    """Receive JSON messages from AssemblyAI and store formatted transcripts."""
    global ws_connection
    while not stop_event.is_set():
        try:
            ws_connection.settimeout(1.0)
            message = ws_connection.recv()
            if not message:
                continue

            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "Begin":
                print(f"[WS-RECV] Session began: ID={data.get('id')}", file=sys.stderr)

            elif msg_type == "Turn":
                transcript_text = data.get("transcript", "")
                formatted = data.get("turn_is_formatted", False)
                end_of_turn = data.get("end_of_turn", False)
                print(f"[WS-RECV] Turn: formatted={formatted}, eot={end_of_turn}, text='{transcript_text[:80]}'", file=sys.stderr)
                if formatted and transcript_text.strip():
                    with transcript_lock:
                        transcript_list.append(transcript_text)

            elif msg_type == "Termination":
                print(f"[WS-RECV] Terminated: audio={data.get('audio_duration_seconds', 0)}s", file=sys.stderr)
                break
            else:
                print(f"[WS-RECV] Unknown: {msg_type}", file=sys.stderr)

        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[WS-RECV] Error: {e}", file=sys.stderr)
            break


def start_websocket():
    """Open WebSocket to AssemblyAI and start sender + receiver threads."""
    global ws_connection, ws_sender_thread, ws_receiver_thread, _pcm_accumulator

    stop_event.clear()
    transcript_list.clear()
    _pcm_accumulator = bytearray()

    # Drain any stale audio from the queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    print(f"[WS] Connecting to {API_ENDPOINT}", file=sys.stderr)
    ws_connection = websocket.create_connection(
        API_ENDPOINT,
        header={"Authorization": YOUR_API_KEY},
        enable_multithread=True,
    )
    print("[WS] Connected!", file=sys.stderr)

    ws_sender_thread = threading.Thread(target=ws_sender_loop, daemon=True)
    ws_sender_thread.start()

    ws_receiver_thread = threading.Thread(target=ws_receiver_loop, daemon=True)
    ws_receiver_thread.start()


def stop_websocket():
    """Send Terminate message, close WebSocket, and join threads."""
    global ws_connection, ws_sender_thread, ws_receiver_thread

    stop_event.set()

    if ws_connection:
        try:
            print("[WS] Sending Terminate...", file=sys.stderr)
            ws_connection.send(json.dumps({"type": "Terminate"}))
            time.sleep(2)
        except Exception as e:
            print(f"[WS] Terminate error: {e}", file=sys.stderr)
        try:
            ws_connection.close()
        except Exception:
            pass
        ws_connection = None

    if ws_sender_thread and ws_sender_thread.is_alive():
        ws_sender_thread.join(timeout=2.0)
    if ws_receiver_thread and ws_receiver_thread.is_alive():
        ws_receiver_thread.join(timeout=2.0)

    ws_sender_thread = None
    ws_receiver_thread = None
    print("[WS] Cleaned up.", file=sys.stderr)


# --- LLM Gateway Analysis (disabled - account does not have LeMUR access) ---
# def analyze_with_llm_gateway(text):
#     """Post transcript to AssemblyAI LLM Gateway for coaching analysis."""
#     headers = {
#         "authorization": YOUR_API_KEY,
#         "content-type": "application/json",
#     }
#
#     prompt = (
#         "You are a helpful coach. Provide an analysis of the transcript "
#         "and offer areas to improve with exact quotes. Include no preamble. "
#         "Start with an overall summary then get into the examples with feedback."
#     )
#
#     llm_gateway_data = {
#         "model": "claude-sonnet-4-20250514",
#         "messages": [
#             {"role": "user", "content": f"{prompt}\n\nTranscript: {text}"}
#         ],
#         "max_tokens": 4000,
#     }
#
#     result = requests.post(
#         LLM_GATEWAY_URL,
#         headers=headers,
#         json=llm_gateway_data,
#     )
#     response = result.json()
#     print("LLM Gateway response:", response)
#     if "choices" not in response:
#         raise RuntimeError(f"LLM Gateway error: {response}")
#     return response["choices"][0]["message"]["content"]


# --- Session State (UI-only state) ---
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
# if "analysis_result" not in st.session_state:
#     st.session_state.analysis_result = None
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

# --- Streamlit UI ---
st.set_page_config(page_title="AssemblyAI Streaming STT", layout="wide")
st.title("AssemblyAI Streaming STT")
st.caption("Real-time speech-to-text powered by AssemblyAI")

# Status indicator
status_placeholder = st.empty()

# WebRTC audio capture — SENDONLY with audio_frame_callback (no media player rendered)
webrtc_ctx = webrtc_streamer(
    key="assemblyai-stt",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# Transcript display
st.subheader("Live Transcript")
transcript_placeholder = st.empty()

# # Analysis section (disabled - account does not have LeMUR access)
# st.subheader("Coaching Analysis")
#
# analysis_placeholder = st.empty()
#
# # Display previous analysis if exists
# if st.session_state.analysis_result:
#     analysis_placeholder.markdown(st.session_state.analysis_result)

# --- Handle WebRTC state changes ---
if webrtc_ctx.state.playing:
    if not st.session_state.recording_active:
        try:
            start_websocket()
            st.session_state.recording_active = True
            st.session_state.session_ended = False
            st.session_state.transcripts = []
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}", file=sys.stderr)
            status_placeholder.error(f"Failed to connect to AssemblyAI: {e}")
            st.stop()

    status_placeholder.success("Recording... Speak into your microphone.")

    # Live transcript polling loop
    while webrtc_ctx.state.playing:
        time.sleep(0.5)
        with transcript_lock:
            current = list(transcript_list)
        if current:
            st.session_state.transcripts = current
            transcript_placeholder.markdown("\n\n".join(current))

else:
    if st.session_state.recording_active:
        # Snapshot transcripts into session state before stopping
        with transcript_lock:
            if transcript_list:
                st.session_state.transcripts = list(transcript_list)
        stop_websocket()
        st.session_state.recording_active = False
        st.session_state.session_ended = True

    if st.session_state.session_ended:
        status_placeholder.warning("Recording stopped.")
        current = st.session_state.transcripts
        if current:
            transcript_placeholder.markdown("\n\n".join(current))
            st.download_button(
                "Download Transcript",
                data="\n\n".join(current),
                file_name="transcript.txt",
                mime="text/plain",
            )

        # # LLM Gateway analysis (disabled - account does not have LeMUR access)
        # analyze_clicked = st.button("Analyze with LLM Gateway", disabled=len(current) == 0)
        # if analyze_clicked and current:
        #     full_transcript = "\n".join(current)
        #     status_placeholder.info("Analyzing transcript with LLM Gateway...")
        #     with st.spinner("Analyzing..."):
        #         try:
        #             result = analyze_with_llm_gateway(full_transcript)
        #             st.session_state.analysis_result = result
        #             analysis_placeholder.markdown(result)
        #             status_placeholder.success("Analysis complete!")
        #         except Exception as e:
        #             status_placeholder.error(f"Analysis failed: {e}")
    else:
        status_placeholder.info("Ready. Click START to begin recording.")
