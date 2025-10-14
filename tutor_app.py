# tutor_app.py that only handles the text based chat with AI tutor

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
TUTOR_CONFIG = {
    "Gyan Mitra (Grade 5)": {"persona": "GYAN_MITRA_PERSONA", "title": "🎓 Chat with Gyan Mitra"},
    "Khel-Khel Mein Guru (Grade 2)": {"persona": "KHEL_GURU_PERSONA", "title": "🎲 Play & Learn with Khel-Khel Mein Guru"}
}

# --- Page Configuration ---
st.set_page_config(
    page_title="CBSE AI Tutor",
    page_icon="🤖",
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
- **Math Formatting:** You MUST use LaTeX for all mathematical fractions, symbols, and equations. Enclose all LaTeX in single dollar signs. For example, to show one-third, you must write `$\frac{1}{3}$`.
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
    """Gets a configuration value safely from .env, then Streamlit secrets."""
    value = os.getenv(key_name)
    if value: return value
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception: pass
    return default_value

def handle_prompt(prompt, model, api_key, lesson_context, persona_text):
    """Appends prompt to history, gets a response, and handles displaying it in the UI."""
    if not api_key or not model:
        st.warning("AI model is not ready. Please check your API key in the sidebar.")
        st.stop()
    if not lesson_context:
        st.warning("Please upload the chapter pages (PDF or Images) in the sidebar.")
        st.stop()

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Start a new chat session if one doesn't exist
                if st.session_state.chat_session is None:
                    st.session_state.chat_session = model.start_chat()
                    # Construct the initial prompt with persona and context
                    initial_prompt_parts = [
                        persona_text, "Here is the content from the textbook pages:",
                        *lesson_context, "\n---\n",
                        "Now, start the conversation by responding to the student's first question:", prompt
                    ]
                    response = st.session_state.chat_session.send_message(initial_prompt_parts)
                else:
                    # Send a follow-up message with the lesson context
                    follow_up_prompt_parts = [prompt, *lesson_context]
                    response = st.session_state.chat_session.send_message(follow_up_prompt_parts)

            response_text = response.text
            st.markdown(response_text)
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Initialize Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_session" not in st.session_state: st.session_state.chat_session = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("👨‍🏫 AI Tutor Setup")
    st.markdown("Configure your AI Tutor and provide the lesson context here.")
    
    # API Key Input
    api_key = get_config("GEMINI_API_KEY")
    if api_key:
        st.success("API key loaded successfully!", icon="✅")
    else:
        st.warning("API key not found. Please add it to your .env or secrets.", icon="⚠️")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
    
    # Tutor Selection
    selected_tutor = st.radio("👤 Choose a Tutor:", list(TUTOR_CONFIG.keys()))
    
    # File Uploader
    st.markdown("### 📚 Upload Chapter Pages")
    uploaded_files = st.file_uploader(
        "Upload PDF or Images of the chapter pages:",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    # Process Uploaded Files
    lesson_context = []
    if uploaded_files:
        with st.spinner("Reading files..."):
            text_context = ""
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                        for page in pdf_document:
                            text_context += page.get_text() + "\n\n"
                    else:
                        img = Image.open(file)
                        lesson_context.append(img)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
            if text_context:
                lesson_context.append(text_context)
            st.success(f"Successfully read {len(uploaded_files)} file(s)!", icon="📄")

    # Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.rerun()

# --- Main App Logic ---
st.title(TUTOR_CONFIG[selected_tutor]["title"])

# Select the appropriate persona text based on the user's choice
persona_text = GYAN_MITRA_PERSONA if selected_tutor == "Gyan Mitra (Grade 5)" else KHEL_GURU_PERSONA

# Configure the Generative AI model
model = None
if not api_key:
    st.info("Please provide your Gemini API Key in the sidebar to begin.")
else:
    try:
        genai.configure(api_key=api_key)
        model_name = get_config("GEMINI_MODEL", "gemini-1.5-pro-latest")
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error configuring the AI model: {e}")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handling ---
if prompt := st.chat_input("What would you like to learn today?"):
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Call the main handler to get and display the AI response
    handle_prompt(prompt, model, api_key, lesson_context, persona_text)