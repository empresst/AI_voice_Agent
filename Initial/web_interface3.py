import streamlit as st
import requests
import base64
from io import BytesIO
import json
import logging
import mimetypes
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import re

# Configure logging
logging.basicConfig(level=logging.INFO, filename='streamlit.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Voice Sales Agent", page_icon="📞")

st.title("AI Voice Sales Agent")
st.markdown("Interact with an AI-powered sales agent pitching the AI Mastery Bootcamp. Enter text or upload audio to simulate a customer call.")

# Initialize session state
if "call_id" not in st.session_state:
    st.session_state.call_id = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "call_in_progress" not in st.session_state:
    st.session_state.call_in_progress = False
if "sentiment_trend" not in st.session_state:
    st.session_state.sentiment_trend = [0.0]
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "objection_count" not in st.session_state:
    st.session_state.objection_count = 0
if "input_method" not in st.session_state:
    st.session_state.input_method = "Text"

# Start Call Form
with st.form("start_call"):
    st.write("### Start a New Call")
    phone_number = st.text_input("Phone Number", placeholder="e.g., 123-456-7890")
    customer_name = st.text_input("Customer Name", placeholder="e.g., John Doe")
    pitch_variant = st.selectbox("Pitch Variant", ["A", "B"])
    start_button = st.form_submit_button("Start Call")

    if start_button and phone_number and customer_name:
        try:
            response = requests.post(
                "http://localhost:8000/start-call",
                json={
                    "phone_number": phone_number,
                    "customer_name": customer_name,
                    "pitch_variant": pitch_variant
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.call_id = data["call_id"]
            st.session_state.conversation.append({"role": "agent", "message": data["first_message"]})
            st.session_state.call_in_progress = True
            st.session_state.turn_count = 1
            audio_bytes = base64.b64decode(data["audio"])
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Call started successfully!")
            logger.info(f"Started call {data['call_id']} for {customer_name}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error starting call: {str(e)}. Ensure the FastAPI server is running on http://localhost:8000.")
            logger.error(f"Start call failed: {str(e)}")

# Respond Section
if st.session_state.call_in_progress:
    st.write("### Respond to Customer")
    input_method = st.radio("Input Method", ["Text", "Audio"], key="input_method_radio")
    st.session_state.input_method = input_method

    customer_message = st.text_input(
        "Customer Message (optional for Audio mode)",
        placeholder="Type customer's response or leave blank for audio...",
        key="customer_message"
    )
    logger.info(f"Customer message entered: {customer_message}")

    audio_input = None
    if input_method == "Audio":
        audio_input = st.file_uploader("Upload Audio Response (WAV/MP3, <10s, <5MB)", type=["wav", "mp3"], key="audio_upload")
        if audio_input:
            st.info("Audio file selected. Click 'Send Response' to submit.")
        else:
            st.warning("Please select a WAV or MP3 file.")

    with st.form("respond"):
        submit_button = st.form_submit_button("Send Response")

        if submit_button:
            if input_method == "Audio" and not audio_input and not customer_message:
                st.error("Please upload an audio file or enter a text message.")
                logger.error("No audio file or message provided for response")
                st.stop()
            if not customer_message and not audio_input:
                st.error("Please provide a text message or upload an audio file.")
                logger.error("No input provided for response")
                st.stop()
            try:
                payload = {"message": customer_message or "Audio input provided" if audio_input else customer_message}
                if audio_input:
                    audio_bytes = audio_input.read()
                    logger.info(f"Audio bytes size: {len(audio_bytes)}, first_20: {audio_bytes[:20].hex()}")
                    if len(audio_bytes) > 5 * 1024 * 1024:
                        st.error("Audio file too large. Please upload a file under 5MB.")
                        logger.error(f"Audio file too large: {len(audio_bytes)} bytes")
                        st.stop()
                    mime_type, _ = mimetypes.guess_type(audio_input.name)
                    if mime_type not in ["audio/wav", "audio/mpeg"]:
                        st.error("Invalid file format. Please upload a WAV or MP3 file.")
                        logger.error(f"Invalid file format: {mime_type}")
                        st.stop()
                    audio_io = BytesIO(audio_bytes)
                    try:
                        if mime_type == "audio/mpeg":
                            audio = MP3(audio_io)
                            duration = audio.info.length
                            bitrate = audio.info.bitrate
                        else:
                            audio = WAVE(audio_io)
                            duration = audio.info.length
                            bitrate = audio.info.sample_rate
                        if duration > 10:
                            st.error("Audio file too long. Please upload a file under 10 seconds.")
                            logger.error(f"Audio file too long: {duration} seconds")
                            st.stop()
                        logger.info(f"Audio validated: name={audio_input.name}, size={len(audio_bytes)} bytes, duration={duration}s, bitrate={bitrate}")
                    except Exception as e:
                        st.error("Invalid or corrupted audio file. Please upload a valid WAV or MP3.")
                        logger.error(f"Audio validation failed: {str(e)}")
                        st.stop()
                    try:
                        # Pre-process bytes to remove invalid characters
                        audio_bytes = bytes([b for b in audio_bytes if 0 <= b <= 255])
                        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                        if not re.match(r'^[A-Za-z0-9+/=]+$', audio_base64):
                            raise ValueError("Invalid base64 characters")
                        base64.b64decode(audio_base64, validate=True)
                        payload["audio"] = audio_base64
                        logger.info(f"Audio encoded, base64 length: {len(audio_base64)}, first_50: {audio_base64[:50]}, last_50: {audio_base64[-50:]}")
                    except (base64.binascii.Error, ValueError) as e:
                        st.warning(f"Audio encoding failed: {str(e)}. Using text input instead. Please provide a text message or try a WAV file.")
                        logger.error(f"Audio encoding failed: {str(e)}, falling back to message")
                        payload["audio"] = None
                
                logger.info(f"Payload prepared: {json.dumps(payload, indent=2)}")
                logger.info(f"Sending payload: message={payload['message']}, audio_length={len(payload.get('audio', '')) if payload.get('audio') is not None else 0}")
                response = requests.post(
                    f"http://localhost:8000/respond/{st.session_state.call_id}",
                    json=payload,
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.conversation.append({"role": "customer", "message": customer_message or "Audio input"})
                st.session_state.conversation.append({"role": "agent", "message": data["reply"]})
                st.session_state.sentiment_trend.append(1.0 if data["sentiment"] == "POSITIVE" else -1.0 if data["sentiment"] == "NEGATIVE" else 0.0)
                audio_bytes = base64.b64decode(data["audio"])
                st.audio(audio_bytes, format="audio/mp3")
                st.markdown(f"**Sentiment**: {data['sentiment']}")
                
                conv_response = requests.get(f"http://localhost:8000/conversation/{st.session_state.call_id}")
                conv_response.raise_for_status()
                conv_data = conv_response.json()
                st.session_state.turn_count = conv_data["turn_count"]
                st.session_state.objection_count = conv_data["objection_count"]
                
                if data["should_end_call"]:
                    st.session_state.call_in_progress = False
                    st.success("Call ended.")
                    logger.info(f"Call {st.session_state.call_id} ended")
            except requests.exceptions.HTTPError as e:
                error_detail = e.response.json().get("detail", str(e))
                st.error(f"Error responding: {error_detail}. Check audio file or server logs.")
                logger.error(f"Respond failed: {error_detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error responding: {str(e)}. Ensure the server is running and inputs are valid.")
                logger.error(f"Respond failed: {str(e)}")

# Display Conversation History
if st.session_state.conversation:
    st.write("### Conversation History")
    for msg in st.session_state.conversation:
        if msg["role"] == "agent":
            st.markdown(f"<p style='color: #2E7D32;'><b>Agent</b>: {msg['message']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color: #1976D2;'><b>Customer</b>: {msg['message']}</p>", unsafe_allow_html=True)

# Sentiment Trend Chart
if st.session_state.sentiment_trend and len(st.session_state.sentiment_trend) > 1:
    st.write("### Sentiment Trend")
    chart_data = {
        "type": "line",
        "data": {
            "labels": [f"Turn {i+1}" for i in range(len(st.session_state.sentiment_trend))],
            "datasets": [{
                "label": "Customer Sentiment",
                "data": st.session_state.sentiment_trend,
                "borderColor": "#4CAF50",
                "backgroundColor": "rgba(76, 175, 80, 0.2)",
                "fill": True
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {"display": True, "text": "Sentiment Score"},
                    "ticks": {"stepSize": 1}
                },
                "x": {
                    "title": {"display": True, "text": "Conversation Turn"}
                }
            }
        }
    }
    st.components.v1.html(f"""
        <canvas id="sentimentChart"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
        <script>
            const ctx = document.getElementById('sentimentChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_data)});
        </script>
    """, height=300)

# Analytics Section
if st.session_state.call_id:
    st.write("### Conversation Analytics")
    st.markdown(f"- **Total Turns**: {st.session_state.turn_count}")
    st.markdown(f"- **Objections Raised**: {st.session_state.objection_count}")