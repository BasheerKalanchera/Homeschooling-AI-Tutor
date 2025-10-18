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
import streamlit.components.v1 as components


def scroll_chat_to_latest(delay_ms: int = 250):
    """
    Scroll Streamlit chat area to latest message/audio.
    Runs JS inside a small component iframe which then uses window.parent to locate
    the Streamlit app DOM. Retries & waits briefly to handle late-rendered audio elements.
    """
    js = f"""
    <script>
    (function() {{
        function tryScroll() {{
            // Try a list of likely scrollable containers (fallbacks)
            const selectors = [
                'main', 
                'main > div[data-testid="stAppViewContainer"]',
                'main section.block-container',
                'div[data-testid="stVerticalBlock"]',
                'div[data-testid="stMainContent"]',
                'section.main'
            ];
            let container = null;
            for (const s of selectors) {{
                try {{
                    container = window.parent.document.querySelector(s);
                    if (container) break;
                }} catch(e){{ }}
            }}

            // If we found a container, scroll it to the bottom smoothly.
            if (container) {{
                container.scrollTo({{ top: container.scrollHeight, behavior: 'smooth' }});
            }}

            // Try to scroll the last audio element into view (center it).
            try {{
                const audios = window.parent.document.querySelectorAll('audio');
                if (audios && audios.length) {{
                    const lastAudio = audios[audios.length - 1];
                    lastAudio.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }} catch(e){{}}

        }}

        // Delay a bit to let Streamlit render the audio element; retry once.
        setTimeout(tryScroll, {delay_ms});
        setTimeout(tryScroll, {delay_ms + 300});
    }})();
    </script>
    """
    components.html(js, height=0)

# --- One-time Setup (per suggestion 1) ---
if "initialized" not in st.session_state:
    # --- Load Environment Variables ---
    load_dotenv()

    # --- One-time NLTK Downloader ---
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        st.info("First-time setup: Downloading NLTK sentence tokenizers...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    
    st.session_state.initialized = True

# --- Helper Function to Sanitize Text for Speech ---
def sanitize_text_for_speech(text):
    """Cleans text for TTS by removing unwanted characters, asterisks, emojis, and converting LaTeX."""
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA70-\U0001FAFF" # symbols and pictographs extended-A 
        "\u2600-\u26FF"  # miscellaneous symbols
        "\u2700-\u27BF"  # dingbats
        "\uFE0F"  # variation selector
        "\u200d"  # zero width joiner
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

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
Your goal is to help him understand concepts from the School textbook pages provided as images or text.

**Your personality and rules:**
- **Tone:** Be cheerful and use simple analogies related to everyday life, or popular Indian and/or Kerala culture.
- **Method:** Use the Socratic method. Ask guiding questions to help the student arrive at the answer himself. Never give the direct answer to a problem unless he is completely stuck.
- **Language:** You MUST use simple English. Use words like 'Great!' or 'Good Job!' for encouragement. Do NOT use words from any other language.
- **Curriculum:** You MUST base all your explanations, examples, and questions strictly on the content visible in the uploaded textbook pages. Do not introduce concepts outside this provided material.
- **VERY IMPORTANT MATH FORMATTING RULE:**
    - You MUST use LaTeX for all mathematical expressions.
    - You MUST enclose the ENTIRE LaTeX expression in single dollar signs (`$`).
    - To write a fraction, you MUST use the exact syntax `\frac{numerator}{denominator}`.
    - A correct example: `$\frac{2}{4} = \frac{1}{2}$`.
- **Interaction:** Start every new session by greeting the student with a very short introduction.
"""
KHEL_GURU_PERSONA = """
You are 'Khel-Khel Mein Guru,' an enthusiastic and fun AI teacher.
Your student is a 7-year-old in 2nd Grade, studying the CBSE curriculum.

**Your personality and rules:**
- **Tone:** Be very enthusiastic, use simple words, short sentences, and a conversational, friendly tone. Use lots of positive reinforcement ('Amazing!', 'You're a star!', 'Wow!').
- **Method:** Teach through interactive challenges and simple stories based on the uploaded textbook pages. Explain a concept clearly and then immediately follow up with a fun activity.
- **Language:** You MUST use extremely simple English, suitable for a 7-year-old. Use words like 'Great!' or 'Good Job!' for encouragement. Do NOT use words from any other language.
- **Curriculum:** You MUST base all your explanations, stories, and activities strictly on the concepts found in the uploaded images or text from the textbook.
- **Interaction:** Start every new session by greeting the student with a fun learning game.

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

# --- Cached Model Loader (per suggestion 6) ---
@st.cache_resource
def get_model(api_key):
    """Loads and configures the GenerativeModel, caching it."""
    genai.configure(api_key=api_key)
    model_name = get_config("GEMINI_MODEL", "gemini-1.5-pro-latest")
    return genai.GenerativeModel(model_name)

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

# --- NEW: File Processing Callback ---
def process_files():
    """
    Callback function to process uploaded files.
    Runs ONLY when the file uploader's state changes.
    """
    # Get files from session state using the uploader's key
    uploaded_files = st.session_state.file_uploader_key
    
    if not uploaded_files:
        st.session_state.lesson_context = [] # Clear context if no files
        return

    new_context = []
    text_context = ""
    
    with st.spinner("Reading files..."):
        for file in uploaded_files:
            try:
                if file.type == "application/pdf":
                    # Read PDF content
                    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
                    for page in pdf_doc:
                        text_context += page.get_text() + "\n\n"
                else:
                    # Read image content
                    new_context.append(Image.open(file))
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
    
    if text_context:
        new_context.append(text_context)
    
    # Store the processed content in session state
    st.session_state.lesson_context = new_context


# --- OPTIMIZED handle_prompt() with 2-call TTS logic ---
def handle_prompt(prompt, model, api_key, lesson_context, persona_text, tts_lang_code, subject, language_key):
    try:
        # Assistant's response container
        with st.chat_message("assistant"):
            # Assistant's thinking spinner
            with st.spinner("Thinking..."):
                if st.session_state.chat_session is None:
                    st.session_state.chat_session = model.start_chat()
                    initial_prompt_parts = [
                        persona_text, "Here is the textbook content:", *lesson_context,
                        "\n---\n", "Respond to the student's first question:", prompt,
                    ]
                    response = st.session_state.chat_session.send_message(initial_prompt_parts)
                else:
                    response = st.session_state.chat_session.send_message([prompt, *lesson_context])

            response_text = response.text.replace('\f', '\\').replace('\\rac', '\\frac')

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            scroll_chat_to_latest(delay_ms=80)

        # Generate and play the audio for the response
        speech_friendly_text = sanitize_text_for_speech(response_text)
        sentences = nltk.sent_tokenize(speech_friendly_text)
        if not sentences:
            return

        FIRST_CHUNK_SIZE = 3
        first_chunk_sentences = sentences[:FIRST_CHUNK_SIZE]
        remaining_sentences = sentences[FIRST_CHUNK_SIZE:]
        
        use_indian_accent = (language_key == "English (India)" and subject in ["Hindi", "Malayalam"])
        tld = 'co.in' if use_indian_accent else 'com'
        
        first_chunk_audio = AudioSegment.empty()
        
        # --- OPTIMIZATION 1: Call gTTS once for the first chunk ---
        if first_chunk_sentences:
            try:
                first_chunk_text = " ".join(first_chunk_sentences)
                mp3_fp = io.BytesIO()
                # Add cache-busting comment
                unique_text = f"{first_chunk_text.strip()}"
                tts = gTTS(text=unique_text, lang=tts_lang_code, tld=tld)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                first_chunk_audio = AudioSegment.from_mp3(mp3_fp)
            except Exception as e:
                print(f"gTTS error (Chunk 1): {e}")

        duration_in_seconds = len(first_chunk_audio) / 1000.0
        audio_placeholder_1 = st.empty()
        audio_placeholder_2 = st.empty()
        rest_audio_container = {}

        # --- OPTIMIZATION 2: Call gTTS once for the remaining chunk in a thread ---
        def generate_rest_audio(sentences, lang_code, result_container, tld):
            rest_audio = AudioSegment.empty()
            if sentences:
                try:
                    remaining_text = " ".join(sentences)
                    mp3_fp = io.BytesIO()
                    # Add cache-busting comment
                    unique_text = f"{remaining_text.strip()}"
                    tts = gTTS(text=unique_text, lang=lang_code, tld=tld)
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)
                    rest_audio = AudioSegment.from_mp3(mp3_fp)
                except Exception as e:
                    print(f"gTTS error in thread (Chunk 2): {e}")
            result_container["audio"] = rest_audio

        if remaining_sentences:
            bg_thread = threading.Thread(
                target=generate_rest_audio,
                args=(remaining_sentences, tts_lang_code, rest_audio_container, tld),
                daemon=True
            )
            bg_thread.start()

        if len(first_chunk_audio) > 0:
            first_buffer = io.BytesIO()
            first_chunk_audio.export(first_buffer, format="mp3")
            first_buffer.seek(0)
            audio_placeholder_1.audio(first_buffer, format='audio/mp3', autoplay=True)
            #scroll_chat_to_latest(delay_ms=250)
            
        # This loop allows the background thread to work while the first audio plays
        elapsed = 0
        sleep_interval = 0.25
        while elapsed < duration_in_seconds:
            time.sleep(sleep_interval)
            elapsed += sleep_interval

        if remaining_sentences:
            bg_thread.join() # Wait for the thread to finish
            rest_audio = rest_audio_container.get("audio", None)
            if rest_audio and len(rest_audio) > 0:
                rest_buffer = io.BytesIO()
                rest_audio.export(rest_buffer, format="mp3")
                rest_buffer.seek(0)
                audio_placeholder_2.audio(rest_buffer, format='audio/mp3', autoplay=True)
                scroll_chat_to_latest(delay_ms=150)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# --- Initialize Session State ---
st.session_state.setdefault('messages', [])
st.session_state.setdefault('chat_session', None)
st.session_state.setdefault('last_audio_processed', None) # To prevent loop
st.session_state.setdefault('lesson_context', []) # NEW: For processed files

with st.sidebar:
    st.title("👨‍🏫 AI Tutor Setup")
    api_key = get_config("GEMINI_API_KEY")
    if api_key: st.success("API key loaded!", icon="✅")
    else:
        st.warning("API key not found.", icon="⚠️")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")

    selected_language = st.selectbox("🌐 Choose your language:", options=list(SUPPORTED_LANGUAGES.keys()), key="selected_language")
    selected_tutor = st.radio("👤 Choose a Tutor:", list(TUTOR_CONFIG.keys()))
    
    subjects = ["English", "Grammar", "EVS", "Mathematics", "Hindi", "Malayalam"]
    selected_subject = st.selectbox("📘 Choose your subject:", options=subjects)

    st.markdown("### 📚 Upload Chapter Pages")
    
    # --- MODIFIED: File Uploader ---
    # We use a key and an on_change callback.
    uploaded_files = st.file_uploader(
        "Upload PDF or Images:", 
        type=['pdf', 'png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key="file_uploader_key",  # A unique key to access its state
        on_change=process_files   # The function to call when files change
    )

    # This part now just shows a success message if context exists
    # It does NOT re-process the files.
    if st.session_state.lesson_context:
        # Get the file count from the uploader's state key
        file_count = len(st.session_state.file_uploader_key)
        st.success(f"Read {file_count} file(s)!", icon="📄")


    # --- MODIFIED: Clear Chat Button ---
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.session_state.last_audio_processed = None
        st.session_state.lesson_context = [] # Clear processed files
        #st.session_state.file_uploader_key = [] # Clear the file uploader widget
        st.rerun()

# --- Main App Logic ---
st.title(TUTOR_CONFIG[selected_tutor]["title"])

base_persona = GYAN_MITRA_PERSONA if selected_tutor == "Gyan Mitra (Grade 5)" else KHEL_GURU_PERSONA
language_instruction = f"\n- **Primary Language:** Your language MUST be {selected_language}."
persona_text = base_persona + language_instruction
tts_lang_code = TTS_LANG_CODES[selected_language]

# --- Load Model (using cached function from suggestion 6) ---
model = None
if api_key:
    try:
        model = get_model(api_key)
    except Exception as e:
        st.error(f"Error configuring AI model: {e}")

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Custom CSS for the sticky footer ---
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    position: fixed;
    bottom: 1rem;
    left: 0;
    right: 0;
    width: 100%;
    background: var(--background-color);
    padding: 10px 1rem; /* Add horizontal padding */
    box-sizing: border-box; /* Include padding in the width */
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)

# Layout for the audiorecorder to be on the right
left_spacer, right_column = st.columns([5, 2])
with right_column:
    # Use a unique key for the audiorecorder to help manage state
    audio_segment_data = audiorecorder("Click to record 🎙️", "Recording... 🔴", key="recorder")

# This is the stable, non-looping interaction block
if audio_segment_data:
    # Check if this audio has already been processed using its raw data
    if st.session_state.last_audio_processed != audio_segment_data.raw_data:
        # It's new audio, process it
        st.session_state.last_audio_processed = audio_segment_data.raw_data # Mark as processed
        
        transcribed_prompt = process_voice_input(audio_segment_data, selected_language)
        if transcribed_prompt:
            st.session_state.messages.append({"role": "user", "content": transcribed_prompt})
            with st.chat_message("user"):
                st.markdown(transcribed_prompt)
            
            if model:
                # --- MODIFIED: Read lesson_context from session state ---
                lesson_context = st.session_state.lesson_context
                
                handle_prompt(
                    transcribed_prompt,
                    model,
                    api_key,
                    lesson_context, # Pass the pre-processed context
                    persona_text,
                    tts_lang_code,
                    selected_subject,
                    selected_language
                )
            else:
                st.warning("Please provide your API key in the sidebar to start.")

# Add a spacer to push the chat history up from the sticky footer
st.markdown("<div style='margin-bottom: 7rem;'></div>", unsafe_allow_html=True)