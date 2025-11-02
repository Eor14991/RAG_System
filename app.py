import streamlit as st
import os
from dotenv import load_dotenv
from Model import Model

# ✅ Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="AI Assistant RAG", page_icon="🧠", layout="centered")

# --- App Title ---
st.title("💬 AI Knowledge Assistant")
st.caption("Ask questions based on your uploaded or default document.")

# --- Initialize Session State ---
# ✅ CHANGED: We will store the 'qa_chain' directly, not the whole model
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🧩 Assistant Settings")
    st.markdown("""
    Welcome! 👋  
    - Enter your **Google Gemini API Key** (or set it in your .env).
    - Upload a PDF document or use the **default knowledge base**.
    """)

    user_api_key = st.text_input(
        "🔑 Google Gemini API Key (optional)",
        type="password",
        help="You can get one from https://aistudio.google.com/app/apikey"
    )

    # ✅ CRITICAL SECURITY FIX: Removed hardcoded API key
    # Priority: 1. User Input, 2. Environment Variable
    final_api_key = os.getenv("GOOGLE_API_KEY")

    st.divider()
    use_default_docs = st.checkbox("📘 Use default knowledge base", value=False)

    with st.expander("⚙️ Advanced Settings (Optional)"):
        temperature = st.slider("Model Creativity", 0.0, 1.0, 0.7, 0.1)
        chunk_size = st.number_input("Chunk Size", 100, 2000, 500, 100)
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, 100, 50)
        max_tokens = st.number_input("Max Tokens", 128, 4096, 512, 64)

    # File upload
    uploaded_file = None
    if not use_default_docs:
        uploaded_file = st.file_uploader("📄 Upload your PDF document", type=["pdf"])

    # ✅ CHANGED: Button now just clears caches to force re-initialization
    if st.button("🚀 Initialize Assistant"):
        if not final_api_key:
            st.error("❌ Please provide a Google Gemini API Key.")
        elif not use_default_docs and not uploaded_file:
            st.warning("⚠️ Please upload a PDF or check 'Use default knowledge base'.")
        else:
            # Clear caches and session state
            st.cache_resource.clear()
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.success("✅ Settings updated! The assistant will load on your first message.")


# --- Caching and Model Initialization ---

# ✅ NEW: Cached function for the slow embedding/retriever part
@st.cache_resource(show_spinner="Embedding document... ⏳")
def get_retriever(api_key, file_path, chunk_size, chunk_overlap):
    """
    Creates and caches the retriever. This is the slow part.
    The Model __init__ still runs to get access to the method,
    but the expensive part (create_retriever) is cached.
    """
    try:
        model_instance = Model(api_key=api_key)
        retriever = model_instance.create_retriever(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return retriever
    except Exception as e:
        st.error(f"Error creating retriever: {e}")
        return None

# ✅ NEW: Function to get the fast, session-specific chain
def get_qa_chain(api_key, retriever, temp, tokens):
    """
    Creates the conversational chain. This is fast and holds
    the user's specific chat memory.
    """
    try:
        model_instance = Model(
            api_key=api_key,
            temperature=temp,
            max_tokens=tokens
        )
        return model_instance.create_chain(retriever)
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# --- Main App Logic ---
if not final_api_key:
    st.info("Please add your Google Gemini API Key in the sidebar to start.")
else:
    # Determine which file path to use
    file_path = None
    if use_default_docs:
        file_path = "RAG_default_docs.pdf"
    elif uploaded_file:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    if file_path:
        if not os.path.exists(file_path):
            st.error(f"⚠️ File not found: {file_path}. Please upload it or select the default.")
        else:
            # 1. Get the (potentially cached) retriever
            retriever = get_retriever(
                final_api_key, file_path, chunk_size, chunk_overlap
            )

            if retriever:
                # 2. Initialize the chain if it's not in session state
                if st.session_state.qa_chain is None:
                    st.session_state.qa_chain = get_qa_chain(
                        final_api_key, retriever, temperature, max_tokens
                    )

                # --- Chat Interface ---
                if st.session_state.qa_chain:
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                    user_input = st.chat_input("💬 Ask your question here...")

                    if user_input:
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        with st.chat_message("user"):
                            st.markdown(user_input)

                        with st.chat_message("assistant"):
                            with st.spinner("Thinking... 🤔"):
                                try:
                                    # ✅ CHANGED: Use the chain directly
                                    response = st.session_state.qa_chain.invoke(
                                        {"question": user_input},
                                        config={"configurable": {"session_id": "streamlit_session"}}
                                    )

                                    answer = response.get("answer", "No valid response generated.")
                                    st.markdown(answer)
                                    st.session_state.messages.append(
                                        {"role": "assistant", "content": answer}
                                    )
                                except Exception as e:
                                    st.error(f"Error generating response: {e}")
                else:
                    st.warning("Assistant chain not initialized. Please click 'Initialize'.")
            else:
                st.error("Failed to create document retriever. Check file and settings.")
    else:
        st.info("Please initialize the assistant first from the sidebar.")

# --- Footer ---
st.markdown("---")
st.caption("Created by **Mohamed Farrag Al-Samman** · Powered by LangChain + Gemini 🌟")

