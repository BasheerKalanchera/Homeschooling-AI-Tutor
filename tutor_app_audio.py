# tutor_app_audio.py

# --- UPDATED SCRIPT ---
# 1. Personas updated to request dual output (display_text vs. speech_text)
# 2. handle_prompt updated to parse this dual output.
# 3. sanitize_text_for_speech updated to be a robust fallback.

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

# --- NEW: Added pytesseract for OCR ---
import pytesseract

# ---
# CRITICAL: TESSERACT INSTALLATION REQUIRED
#
# This script now uses pytesseract for Optical Character Recognition (OCR).
# You MUST install the Tesseract engine on your system:
#
# 1. Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
#    (Make sure to check the "Malayalam" language pack during installation)
# 2. macOS: brew install tesseract tesseract-lang
# 3. Linux: sudo apt-get install tesseract-ocr tesseract-ocr-mal
#
# You also need the Python library: pip install pytesseract
# ---


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

# --- Helper Function to Sanitize Text for Speech (NOW A FALLBACK) ---
def sanitize_text_for_speech(text):
    """
    Cleans text for TTS by removing unwanted characters, asterisks, emojis, 
    and converting LaTeX.
    This is now a FALLBACK for when the LLM fails to provide clean speech_text.
    """

    # 1. Remove LaTeX math delimiters
    text = text.replace('$', '')

    # 2. Handle LaTeX \text{...} command
    #    Example: "\text{ cm}" becomes " cm"
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)

    # 3. Handle LaTeX fractions
    fraction_pattern = r"\\frac\{(\d+)\}\{(\d+)\}"
    def replace_fraction(match):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if numerator == 1:
            if denominator == 2: return "one-half"
            if denominator == 3: return "one-third"
            if denominator == 4: return "one-fourth"
        return f"{numerator} over {denominator}"
    text = re.sub(fraction_pattern, replace_fraction, text)

    # 4. Handle Roman numerals in parentheses, e.g., (v) -> "roman numeral 5"
    roman_map = {
        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
        'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10'
    }
    def replace_roman(match):
        numeral = match.group(1).lower()
        if numeral in roman_map:
            return f"roman numeral {roman_map[numeral]}"
        return match.group(0) # Return original if not in map
    text = re.sub(r'\(([ivxIVX]+)\)', replace_roman, text)


    # 5. Remove other common LaTeX commands/artifacts
    text = text.replace('\\,', ' ')  # thin space
    text = text.replace('\\ ', ' ')  # escaped space
    text = text.replace('\\', '')    # Remove any remaining stray backslashes

    # 6. Handle "fill in the blank" underscores/dashes
    #    Replaces "____" or "----" with the word "blank"
    text = re.sub(r'[_—-]{3,}', ' blank ', text) 

    # 7. Original Emoji remover
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
    
    # 8. Original simple replacements
    text = text.replace('*', '')
    text = text.replace('\f', '')  # Remove form feed character

    # 9. Final cleanup: normalize all spaces to a single space
    processed_text = re.sub(r'\s+', ' ', text).strip()
    
    return processed_text

# --- Constants and Page Config ---
TUTOR_CONFIG = {
    "Gyan Mitra (Grade 5)": {"persona": "GYAN_MITRA_PERSONA", "title": "🎓 Chat with Gyan Mitra (Voice)"},
    "Khel-Khel Mein Guru (Grade 2)": {"persona": "KHEL_GURU_PERSONA", "title": "🎲 Play & Learn with Khel-Khel Mein Guru (Voice)"}
}
SUPPORTED_LANGUAGES = {
    "English (India)": "en-IN",
    "Hindi (हिन्दी)": "hi-IN",
    "Malayalam (മലയാളം)": "ml-IN"    
}
TTS_LANG_CODES = {
    "English (India)": "en",
    "Hindi (हिन्दी)": "hi",
    "Malayalam (മലയാളം)": "ml"    
}

st.set_page_config(
    page_title="CBSE AI Voice Tutor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Prompts and Personas ---

# --- UPDATED PERSONA 1 ---
GYAN_MITRA_PERSONA = """
You are 'Gyan Mitra,' a friendly, patient, and encouraging AI tutor.
Your student is a 10-year-old in 5th Grade, following the CBSE curriculum in India.
Your goal is to help him understand concepts from the School textbook pages provided as images or text.

**Your personality and rules:**
- **Tone:** Be cheerful and use simple analogies related to everyday life, or popular Indian and/or Kerala culture.
- **Method:** Use the Socratic method. Ask guiding questions to help the student arrive at the answer himself. Never give the direct answer to a problem unless he is completely stuck.
- **Curriculum:** You MUST base all your explanations, examples, and questions strictly on the content visible in the uploaded textbook pages. Do not introduce concepts outside this provided material.
- **Conciseness:** Keep replies concise. Explain one small concept or ask one guiding question at a time.
- **Engagement:** Always try to end your response with a question to keep the conversation going.

**CRITICAL: RESPONSE FORMAT**
You MUST provide your answer in two parts, separated by the exact token "||SPEECH_BREAK||".

1.  **display_text:** This is the visual response for the student.
    - It MUST use LaTeX enclosed in single dollar signs (`$`) for all math (e.g., `$\frac{1}{2}$`).
    - **VERY IMPORTANT:** After writing a LaTeX expression like `$\frac{4}{6}$`, DO NOT add any extra plain-text numbers like `6 4` or `4/6`. The model must ONLY output the pure LaTeX.
    - **Bad display_text:** `Here is $\frac{4}{6} 6 4`
    - **Good display_text:** `Here is $\frac{4}{6}`
    - Use markdown for formatting and underscores (`____`) for "fill-in-the-blank" questions.

2.  **speech_text:** This is the text for the audio engine. It MUST be 100% clean, spelled-out, and ready for Text-to-Speech.
    - All LaTeX (like `$\frac{1}{2}$`) must be converted to plain words (e.g., "one-half").
    - All math (like `$6 \text{ m } 50 \text{ cm}$`) must be plain words (e.g., "6 meters 50 centimeters").
    - All "fill-in-the-blank" lines (`____`) must be replaced with the word "blank".
    - All parenthesized list items like (i), (ii), (v) must be spelled out (e.g., "roman numeral 1", "roman numeral 2", "roman numeral 5").
    - Do not include any asterisks, markdown, or emojis.

**EXAMPLE 1 (Math):**
display_text: Great job! The answer is $\frac{1}{2}$. Now, try this one: 5 ____ 10.
||SPEECH_BREAK||
speech_text: Great job! The answer is one-half. Now, try this one: 5 blank 10.

**EXAMPLE 2 (List):**
display_text: Let's look at question (v), "In shape (v),..."
||SPEECH_BREAK||
speech_text: Let's look at question roman numeral 5, "In shape roman numeral 5,..."
"""

# --- UPDATED PERSONA 2 ---
KHEL_GURU_PERSONA = """
You are 'Khel-Khel Mein Guru,' an enthusiastic and fun AI teacher.
Your student is a 7-year-old in 2nd Grade, studying the CBSE curriculum.

**Your personality and rules:**
- **Tone:** Be very enthusiastic, use simple words, short sentences, and a conversational, friendly tone. Use lots of positive reinforcement ('Amazing!', 'You're a star!', 'Wow!').
- **Method:** Teach through interactive challenges and simple stories based on the uploaded textbook pages.
- **Curriculum:** You MUST base all your explanations, stories, and activities strictly on the concepts found in the uploaded images or text from the textbook.
- **Conciseness:** Keep replies very short (2-3 sentences). Use simple bullet points with emojis (like ✨ or 👉) instead of dense text.
- **Engagement:** Always end every single message with a simple, fun question to keep the student engaged.

**CRITICAL: RESPONSE FORMAT**
You MUST provide your answer in two parts, separated by the exact token "||SPEECH_BREAK||".

1.  **display_text:** This is the visual response for the student.
    - It MUST use LaTeX enclosed in single dollar signs (`$`) for all math (e.g., `$2 + 2 = 4$`).
    - **VERY IMPORTANT:** After writing a LaTeX expression like `$\frac{4}{6}$`, DO NOT add any extra plain-text numbers like `6 4` or `4/6`. The model must ONLY output the pure LaTeX.
    - **Bad display_text:** `Here is $\frac{4}{6} 6 4`
    - **Good display_text:** `Here is $\frac{4}{6}`
    - Use markdown for formatting and underscores (`____`) for "fill-in-the-blank" questions.

2.  **speech_text:** This is the text for the audio engine. It MUST be 100% clean, spelled-out, and ready for Text-to-Speech.
    - All LaTeX (like `$\frac{1}{2}$`) must be converted to plain words (e.g., "one-half").
    - All math (like `$2 + 2 = 4$`) must be plain words (e.g., "2 plus 2 equals 4").
    - All "fill-in-the-blank" lines (`____`) must be replaced with the word "blank".
    - All parenthesized list items like (i), (ii), (v) must be spelled out (e.g., "roman numeral 1", "roman numeral 2", "roman numeral 5").
    - Do not include any asterisks, markdown, or emojis.

**EXAMPLE 1 (Math):**
display_text: You got it! ✨ $2 + 2 = 4$. Now, what is $3 + 1$?
||SPEECH_BREAK||
speech_text: You got it! 2 plus 2 equals 4. Now, what is 3 plus 1?

**EXAMPLE 2 (List):**
display_text: Let's do activity (ii)!
||SPEECH_BREAK||
speech_text: Let's do activity roman numeral 2!
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

# --- PHASE 2 MODIFICATION: START ---
# This function is now the "Intelligent Triage" pipeline
@st.cache_data(max_entries=10) # Cache processing for uploaded files
def process_files_triage(uploaded_files_data, selected_subject, selected_language):
    """
    Callback function to process uploaded files with "Intelligent Triage".
    Detects native vs. scanned PDFs, OCRs images, and builds a multimodal context.
    
    Returns:
        list[dict]: A list of "chunks", where each chunk is a dictionary:
                    {"text": str|None, "image": PIL.Image|None, "source": str}
    """
    new_context = []
    
    # Map subjects/languages to Tesseract language codes
    lang_map = {
        "Malayalam": "mal",
        "Hindi": "hin",
        "English": "eng",
        "Grammar": "eng",
        "EVS": "eng",
        "Mathematics": "eng"        
    }
    
    # Determine the correct OCR language
    ocr_lang_key = selected_subject if selected_subject in lang_map else selected_language
    ocr_lang = lang_map.get(ocr_lang_key, "eng") # Default to English
    
    if ocr_lang != "eng":
        st.sidebar.info(f"Using Tesseract '{ocr_lang}' language pack for OCR.")

    with st.spinner("Analyzing and extracting content from files..."):
        for file_data in uploaded_files_data:
            # --- THIS IS THE FIX ---
            file_data = dict(file_data)
            # --- END FIX ---
            file_name = file_data['name']
            file_type = file_data['type']
            file_bytes = file_data['bytes']
            
            try:
                if file_type == "application/pdf":
                    # --- PDF TRIAGE ---
                    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                    is_scanned = False
                    
                    # Test the first page for text
                    first_page_text = pdf_doc[0].get_text("text")
                    if not first_page_text.strip():
                        is_scanned = True
                        
                    if is_scanned:
                        # --- Strategy 1: Scanned PDF ---
                        st.sidebar.warning(f"{file_name} seems scanned. Processing with OCR.")
                        for page_num, page in enumerate(pdf_doc):
                            # Render page to an image
                            pix = page.get_pixmap(dpi=200) # 200 DPI is a good balance
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # OCR the image
                            page_text = pytesseract.image_to_string(img, lang=ocr_lang)
                            
                            # Add BOTH text and image to the context
                            new_context.append({
                                "source": f"{file_name} (Page {page_num + 1})",
                                "text": page_text,
                                "image": img  # Send the full-res image to the LLM
                            })
                            
                    else:
                        # --- Strategy 2: Native PDF ---
                        st.sidebar.success(f"{file_name} is a native PDF.")
                        for page_num, page in enumerate(pdf_doc):
                            # Add text chunk
                            page_text = page.get_text("text")
                            if page_text.strip():
                                new_context.append({
                                    "source": f"{file_name} (Page {page_num + 1} Text)",
                                    "text": page_text,
                                    "image": None
                                })
                            
                            # Also extract images from the page as separate chunks
                            image_list = page.get_images(full=True)
                            for img_index, img_info in enumerate(image_list):
                                try:
                                    xref = img_info[0]
                                    base_image = pdf_doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    img = Image.open(io.BytesIO(image_bytes))
                                    
                                    new_context.append({
                                        "source": f"{file_name} (Page {page_num + 1}, Image {img_index + 1})",
                                        "text": None, # No text for this chunk, just the image
                                        "image": img
                                    })
                                except Exception as img_e:
                                    # Skip images that error out (e.g., small masks)
                                    print(f"Skipping non-renderable image on page {page_num+1}: {img_e}")

                else:
                    # --- Strategy 3: Image File ---
                    img = Image.open(io.BytesIO(file_bytes))
                    
                    # OCR the image
                    extracted_text = pytesseract.image_to_string(img, lang=ocr_lang)
                    
                    # Add BOTH text and image to the context
                    new_context.append({
                        "source": f"{file_name}",
                        "text": extracted_text,
                        "image": img
                    })

            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
    
    return new_context

def run_file_processing():
    """
    Wrapper function to be called by the `on_change` event.
    It reads file data and passes it to the cached triage function.
    """
    uploaded_files = st.session_state.file_uploader_key
    if not uploaded_files:
        st.session_state.lesson_context = []
        return
        
    # Create a serializable list of file data for caching
    uploaded_files_data = [
        {"name": f.name, "type": f.type, "bytes": f.read()} 
        for f in uploaded_files
    ]
    
    # Call the cached function
    st.session_state.lesson_context = process_files_triage(
        tuple(frozenset(item.items()) for item in uploaded_files_data), # Make it hashable
        st.session_state.selected_subject,
        st.session_state.selected_language
    )
# --- PHASE 2 MODIFICATION: END ---


# --- ERROR FIX: START ---
def clear_all_state():
    """Callback function to clear all session state."""
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.last_audio_processed = None
    #st.session_state.lesson_context = [] # Clear processed files
    
    # Set the file uploader's state to an empty list.
    # This is allowed because it's in a callback.
    #st.session_state.file_uploader_key = [] 
    
    # Clear the cache for file processing
    #process_files_triage.clear()
    # Streamlit reruns automatically after the callback, no st.rerun() needed.
    # --- THIS IS THE FIX ---
    # We must also clear the state of the recorder widget itself.
    # Its key is "recorder"
    if "recorder" in st.session_state:
        st.session_state.recorder = None
    # --- END FIX ---
# --- ERROR FIX: END ---


# --- PHASE 2 MODIFICATION: START ---
# --- UPDATED handle_prompt function ---
def handle_prompt(prompt, model, api_key, lesson_context, persona_text, tts_lang_code, subject, language_key):
    try:
        # Assistant's response container
        with st.chat_message("assistant"):
            # Assistant's thinking spinner
            with st.spinner("Thinking..."):
                if st.session_state.chat_session is None:
                    # --- THIS IS THE MULTIMODAL UPGRADE ---
                    st.session_state.chat_session = model.start_chat()
                    
                    # 1. Start with the persona
                    initial_prompt_parts = [
                        persona_text, 
                        "Here is the textbook content you must use:"
                    ]
                    
                    # 2. Add the multimodal context (text and images)
                    if not lesson_context:
                        initial_prompt_parts.append("\n[No textbook content was uploaded. Please answer based on general knowledge.]")
                    else:
                        for chunk in lesson_context:
                            # Add a separator for clarity
                            initial_prompt_parts.append(f"\n--- Context from {chunk['source']} ---")
                            
                            # If this chunk has text, append the text string
                            if chunk["text"] and chunk["text"].strip():
                                initial_prompt_parts.append(chunk["text"])
                                
                            # If this chunk has an image, append the PIL Image object
                            if chunk["image"]:
                                initial_prompt_parts.append(chunk["image"])

                    
                    # 3. Add the user's first prompt at the very end
                    initial_prompt_parts.append("\n---\n")
                    initial_prompt_parts.append("Now, respond to the student's first question:")
                    initial_prompt_parts.append(prompt)
                    
                    # 4. Send the *entire list* of [text, images, text, text, ...]
                    response = st.session_state.chat_session.send_message(initial_prompt_parts)
                
                else:
                    # SUBSEQUENT TURNS: Only send the new prompt. 
                    response = st.session_state.chat_session.send_message(prompt)
                    
            raw_response_text = response.text.replace('\f', '\\').replace('\\rac', '\\frac')

            # --- NEW: Parse the dual output ---
            if "||SPEECH_BREAK||" in raw_response_text:
                parts = raw_response_text.split("||SPEECH_BREAK||", 1)
                display_text = parts[0].replace("display_text:", "").strip()
                speech_text = parts[1].replace("speech_text:", "").strip()
                
                # Handle empty speech_text just in case
                if not speech_text.strip():
                    speech_text = sanitize_text_for_speech(display_text)
                    
            else:
                # FALLBACK: If the model forgets the format, use the old method
                st.warning("Model response format error. Using fallback sanitizer.", icon="⚠️")
                display_text = raw_response_text
                speech_text = sanitize_text_for_speech(display_text)
            # --- END NEW ---

            # 1. Show the pretty visual text to the user
            st.markdown(display_text)
            st.session_state.messages.append({"role": "assistant", "content": display_text})
            scroll_chat_to_latest(delay_ms=80)

        # 2. Generate and play the audio using the CLEAN speech text
        sentences = nltk.sent_tokenize(speech_text)
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
# --- PHASE 2 MODIFICATION: END ---


# --- Initialize Session State ---
st.session_state.setdefault('messages', [])
st.session_state.setdefault('chat_session', None)
st.session_state.setdefault('last_audio_processed', None) # To prevent loop
st.session_state.setdefault('lesson_context', []) # NEW: For processed files (list[dict])

with st.sidebar:
    st.title("👨‍🏫 AI Tutor Setup")
    api_key = get_config("GEMINI_API_KEY")
    if api_key: st.success("API key loaded!", icon="✅")
    else:
        st.warning("API key not found.", icon="⚠️")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")

    selected_language = st.selectbox(
        "🌐 Choose your language:", 
        options=list(SUPPORTED_LANGUAGES.keys()), 
        key="selected_language"
    )
    selected_tutor = st.radio("👤 Choose a Tutor:", list(TUTOR_CONFIG.keys()))
    
    subjects = ["English", "Grammar", "EVS", "Mathematics", "Hindi", "Malayalam"]
    selected_subject = st.selectbox(
        "📘 Choose your subject:", 
        options=subjects, 
        key="selected_subject"
    )

    st.markdown("### 📚 Upload Chapter Pages")
    
    # --- MODIFIED: File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload PDF or Images:", 
        type=['pdf', 'png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key="file_uploader_key",  # A unique key to access its state
        on_change=run_file_processing   # Call the wrapper function
    )

    # This part now just shows a success message if context exists
    if st.session_state.lesson_context:
        chunk_count = len(st.session_state.lesson_context)
        st.success(f"Processed {chunk_count} text/image chunks!", icon="📄")


    # --- ERROR FIX: START ---
    # Replaced the `if st.button(...)` block with this:
    st.button("Clear Chat History", on_click=clear_all_state)
    # --- ERROR FIX: END ---


# --- Main App Logic ---
st.title(TUTOR_CONFIG[selected_tutor]["title"])

# 1. Get the base persona
base_persona = GYAN_MITRA_PERSONA if selected_tutor == "Gyan Mitra (Grade 5)" else KHEL_GURU_PERSONA
tts_lang_code = TTS_LANG_CODES[selected_language]

# 2. Define the dynamic language instruction
language_instruction = f"\n- **Language:** You MUST converse ONLY in {selected_language}. Do not use any other language, except for technical terms if absolutely necessary."

# 3. Define the dynamic subject-specific instruction
subject_instruction = ""
if selected_subject == "Malayalam":
    subject_instruction = (
        "\n- **Subject Focus (Malayalam):** You are a Malayalam language expert. "
        "Your student is learning Malayalam. "
        "When the student asks about the Malayalam textbook (e.g., poems, stories, grammar), "
        "you MUST provide clear, accurate, and culturally relevant explanations *in Malayalam*. "
        "Pay close attention to specific user requests, such as: "
        "  - 'അർത്ഥം' (meaning): Provide the meaning of the word. "
        "  - 'വ്യാകരണം' (grammar): Explain the grammar concept. "
        "  - 'ആശയം' (summary/idea): Summarize the passage or poem. "
        "  - 'ചോദ്യം' (question): Answer the question based on the text. "
        "Always be encouraging and. ensure your explanation is appropriate for the student's grade level."
    )



# 4. Assemble the final prompt
persona_text = base_persona + language_instruction + subject_instruction


# --- Load Model ---
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

if audio_segment_data:
    if st.session_state.last_audio_processed != audio_segment_data.raw_data:
        st.session_state.last_audio_processed = audio_segment_data.raw_data
        
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
                    lesson_context, # Pass the multimodal list[dict] context
                    persona_text,
                    tts_lang_code,
                    selected_subject,
                    selected_language
                )
            else:
                st.warning("Please provide your API key in the sidebar to start.")

# Add a spacer to push the chat history up from the sticky footer
st.markdown("<div style='margin-bottom: 7rem;'></div>", unsafe_allow_html=True)