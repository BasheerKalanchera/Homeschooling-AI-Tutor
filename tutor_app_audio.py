# tutor_app_audio.py

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
from audiorecorder import audiorecorder
import speech_recognition as sr
from pydub import AudioSegment
import io
from gtts import gTTS
import nltk
import time
import re
import threading

# --- Load Environment Variables ---
load_dotenv()

# --- One-time NLTK Downloader ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("First-time setup: Downloading the sentence tokenizer...")
    nltk.download('punkt', quiet=True)


# --- Helper Function to Sanitize Text for Speech ---
def sanitize_text_for_speech(text):
    """Cleans text for TTS by removing unwanted characters, asterisks, and converting LaTeX."""
    text = text.replace('*', '')
    text = text.replace('\f', '')  # Remove form feed character
    pattern = r"\\frac\{(\d+)\}\{(\d+)\}"

    def replace_fraction(match):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if numerator == 1:
            if denominator == 2:
                return "one-half"
            if denominator == 3:
                return "one-third"
            if denominator == 4:
                return "one-fourth"
        return f"{numerator} over {denominator}"

    text = text.replace('$', '')
    processed_text = re.sub(pattern, replace_fraction, text)
    return processed_text


# --- Constants and Page Config ---
TUTOR_CONFIG = {
    "Gyan Mitra (Grade 5)": {"persona": "GYAN_MITRA_PERSONA", "title": "🎓 Chat with Gyan Mitra (Voice)"},
    "Khel-Khel Mein Guru (Grade 2)": {"persona": "KHEL_GURU_PERSONA", "title": "🎲 Play & Learn with Khel-Khel Mein Guru (Voice)"}
}
SUPPORTED_LANGUAGES = {
    "English (India)": "en-IN",
    "Hindi (हिन्दी)": "hi-IN",
    "Malayalam (മലയാളം)": "ml-IN",
    "Arabic (Egypt)": "ar-EG",
    "Arabic (Saudi Arabia)": "ar-SA",
    "Arabic (U.A.E.)": "ar-AE",
}
TTS_LANG_CODES = {
    "English (India)": "en",
    "Hindi (हिन्दी)": "hi",
    "Malayalam (മലയാളം)": "ml",
    "Arabic (Egypt)": "ar",
    "Arabic (Saudi Arabia)": "ar",
    "Arabic (U.A.E.)": "ar",
}

st.set_page_config(
    page_title="CBSE AI Voice Tutor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Prompts and Personas ---
GYAN_MITRA_PERSONA = """
You are 'Gyan Mitra,' a friendly, patient, and encouraging AI tutor.
Your student is a 10-year-old in 5th Grade, following the CBSE curriculum in India.
Your goal is to help him understand concepts from the NCERT textbook pages provided as images or text.

**Your personality and rules:**
- **Tone:** Be cheerful and use simple analogies related to everyday life, cricket, or popular Indian culture.
- **Method:** Use the Socratic method. Ask guiding questions to help him arrive at the answer himself. Never give the direct answer to a problem unless he is completely stuck.
- **Language:** You MUST use simple English. Use words like 'Great!' or 'Good Job!' for encouragement. Do NOT use words from any other language.
- **Curriculum:** You MUST base all your explanations, examples, and questions strictly on the content visible in the uploaded textbook pages. Do not introduce concepts outside this provided material.
- **VERY IMPORTANT MATH FORMATTING RULE:**
    - You MUST use LaTeX for all mathematical expressions.
    - You MUST enclose the ENTIRE LaTeX expression in single dollar signs (`$`).
    - To write a fraction, you MUST use the exact syntax `\frac{numerator}{denominator}`.
    - A correct example: `$\frac{2}{4} = \frac{1}{2}$`.
- **Interaction:** Start every new session by greeting the student warmly.
"""
KHEL_GURU_PERSONA = """
You are 'Khel-Khel Mein Guru,' an enthusiastic and fun AI teacher.
Your student is a 7-year-old in 2nd Grade, studying the CBSE curriculum.

**Your personality and rules:**
- **Tone:** Be very enthusiastic, use simple words, short sentences, and a conversational, friendly tone. Use lots of positive reinforcement ('Amazing!', 'You're a star!', 'Wow!').
- **Method:** Teach through interactive challenges and simple stories based on the uploaded textbook pages. Explain a concept clearly and then immediately follow up with a fun activity.
- **Language:** You MUST use extremely simple English, suitable for a 7-year-old. Use words like 'Great!' or 'Good Job!' for encouragement. Do NOT use words from any other language.
- **Curriculum:** You MUST base all your explanations, stories, and activities strictly on the concepts found in the uploaded images or text from the textbook.
- **Interaction:** Start every new session with an exciting greeting and suggest a fun learning game.

**VERY IMPORTANT RESPONSE STYLE:**
- **Keep it short!** Your replies must be very short and easy to read. Aim for only 2-3 sentences.
- **No long paragraphs.** Use simple bullet points or lists with emojis (like ✨ or 👉) instead of dense text.
- **One idea at a time.** Explain only one small thing in each message.
- **Always ask a question.** End every single message with a simple, fun question to keep the student engaged.
"""


# --- Helper functions ---
def get_config(key_name: str, default_value=None):
    value = os.getenv(key_name)
    if value:
        return value
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return default_value


def process_voice_input(audio_segment, language_key):
    if not audio_segment:
        return None
    try:
        with st.spinner("Transcribing..."):
            r = sr.Recognizer()
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
            language_code = SUPPORTED_LANGUAGES[language_key]
            return r.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio. Please try again.")
    except Exception as e:
        st.error(f"Audio processing error: {e}")
    return None


# --- Improved handle_prompt() with two audio placeholders ---
def handle_prompt(prompt, model, api_key, lesson_context, persona_text, tts_lang_code):
    """
    Uses threaded preload system with two placeholders to avoid
    overwriting first audio before playback finishes.
    """
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.chat_session is None:
                    st.session_state.chat_session = model.start_chat()
                    initial_prompt_parts = [
                        persona_text,
                        "Here is the textbook content:",
                        *lesson_context,
                        "\n---\n",
                        "Respond to the student's first question:",
                        prompt,
                    ]
                    response = st.session_state.chat_session.send_message(initial_prompt_parts)
                else:
                    response = st.session_state.chat_session.send_message([prompt, *lesson_context])

            response_text = response.text
            # ** KEY FIX HERE **
            # Force-correct the malformed fraction character before displaying or speaking.
            response_text = response_text.replace('\f', '\\')
            response_text = response_text.replace('\\rac', '\\frac')
            

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # --- Prepare TTS text ---
        speech_friendly_text = sanitize_text_for_speech(response_text)
        sentences = nltk.sent_tokenize(speech_friendly_text)
        if not sentences:
            return

        FIRST_CHUNK_SIZE = 3
        first_chunk_sentences = sentences[:FIRST_CHUNK_SIZE]
        remaining_sentences = sentences[FIRST_CHUNK_SIZE:]

        # --- Generate first chunk ---
        first_chunk_audio = AudioSegment.empty()
        for sentence in first_chunk_sentences:
            try:
                mp3_fp = io.BytesIO()
                tts = gTTS(text=sentence, lang=tts_lang_code)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                first_chunk_audio += AudioSegment.from_mp3(mp3_fp)
            except Exception:
                pass

        duration_in_seconds = len(first_chunk_audio) / 1000.0

        # --- Prepare placeholders ---
        audio_placeholder_1 = st.empty()
        audio_placeholder_2 = st.empty()

        # --- Start preloading 2nd clip in background ---
        rest_audio_container = {}

        def generate_rest_audio(sentences, lang_code, result_container):
            rest_audio = AudioSegment.empty()
            for sentence in sentences:
                try:
                    mp3_fp = io.BytesIO()
                    tts = gTTS(text=sentence, lang=lang_code)
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)
                    rest_audio += AudioSegment.from_mp3(mp3_fp)
                except Exception:
                    pass
            result_container["audio"] = rest_audio

        if remaining_sentences:
            bg_thread = threading.Thread(
                target=generate_rest_audio,
                args=(remaining_sentences, tts_lang_code, rest_audio_container),
                daemon=True
            )
            bg_thread.start()

        # --- Play first audio ---
        first_buffer = io.BytesIO()
        first_chunk_audio.export(first_buffer, format="mp3")
        first_buffer.seek(0)
        audio_placeholder_1.audio(first_buffer, format='audio/mp3', autoplay=True)

        # --- Wait loop for playback duration ---
        elapsed = 0
        sleep_interval = 0.25
        while elapsed < duration_in_seconds:
            time.sleep(sleep_interval)
            elapsed += sleep_interval

        # --- After first finishes, play second if ready ---
        if remaining_sentences:
            bg_thread.join(timeout=0)  # ensure finished if done
            rest_audio = rest_audio_container.get("audio", None)
            if rest_audio and len(rest_audio) > 0:
                # 🔹 Clear the first audio player from UI
                audio_placeholder_1.empty()

                # Then play the next clip
                rest_buffer = io.BytesIO()
                rest_audio.export(rest_buffer, format="mp3")
                rest_buffer.seek(0)
                audio_placeholder_2.audio(rest_buffer, format='audio/mp3', autoplay=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# --- Initialize Session State and Sidebar ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

with st.sidebar:
    st.title("👨‍🏫 AI Tutor Setup")
    api_key = get_config("GEMINI_API_KEY")
    if api_key:
        st.success("API key loaded!", icon="✅")
    else:
        st.warning("API key not found.", icon="⚠️")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")

    selected_language = st.selectbox("🌐 Choose your language:", options=list(SUPPORTED_LANGUAGES.keys()), key="selected_language")
    selected_tutor = st.radio("👤 Choose a Tutor:", list(TUTOR_CONFIG.keys()))

    st.markdown("### 📚 Upload Chapter Pages")
    uploaded_files = st.file_uploader("Upload PDF or Images:", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

    lesson_context = []
    if uploaded_files:
        with st.spinner("Reading files..."):
            text_context = ""
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
                        for page in pdf_doc:
                            text_context += page.get_text() + "\n\n"
                    else:
                        lesson_context.append(Image.open(file))
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
            if text_context:
                lesson_context.append(text_context)
            st.success(f"Read {len(uploaded_files)} file(s)!", icon="📄")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.rerun()

# --- Main App Logic ---
st.title(TUTOR_CONFIG[selected_tutor]["title"])
st.markdown("Record your question below and the AI Tutor will respond in voice.")

base_persona = GYAN_MITRA_PERSONA if selected_tutor == "Gyan Mitra (Grade 5)" else KHEL_GURU_PERSONA
language_instruction = f"\n- **Primary Language:** Your language MUST be {selected_language}."
persona_text = base_persona + language_instruction
tts_lang_code = TTS_LANG_CODES[selected_language]

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model_name = get_config("GEMINI_MODEL", "gemini-1.5-pro-latest")
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error configuring AI model: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    audio_segment_data = audiorecorder("Click to record 🎙️", "Recording... 🔴")

if audio_segment_data:
    transcribed_prompt = process_voice_input(audio_segment_data, selected_language)
    if transcribed_prompt:
        st.session_state.messages.append({"role": "user", "content": transcribed_prompt})
        with st.chat_message("user"):
            st.markdown(transcribed_prompt)
        if model:
            handle_prompt(transcribed_prompt, model, api_key, lesson_context, persona_text, tts_lang_code)
        else:
            st.warning("Please provide your API key in the sidebar to start.")
