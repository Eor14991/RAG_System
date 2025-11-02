from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os


# Load environment variables


# Optional: Safety enums (if available)
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    HarmCategory = HarmBlockThreshold = None


class Model:
    def __init__(self,
                 api_key: str,
                 temperature=0.7,
                 max_tokens=512,
                 model_name: str = "gemini-2.0-flash"):
        """Initialize the AI Model (Gemini + LangChain)."""
        if not api_key:
            raise ValueError("❌ Google Gemini API Key is required. Please set it in your .env file.")

        self.API = api_key.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

        # Set API key for the environment
        os.environ["GOOGLE_API_KEY"] = self.API

        # Build safety settings if available
        safety_settings = None
        if HarmCategory and HarmBlockThreshold:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.API,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            safety_settings=safety_settings
        )

    # -------------------------------
    def create_retriever(self, file_path: str, chunk_size=500, chunk_overlap=100):
        """Load a PDF, split it into chunks, embed it, and create a retriever."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"📄 Document not found: {file_path}")

        print(f"📚 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = splitter.split_documents(docs)

        try:
            embedding = FastEmbedEmbeddings()
        except Exception as e:
            raise ImportError(f"FastEmbed is not installed. Run: pip install fastembed\nError: {e}")

        try:
            vectorstore = Chroma.from_documents(splits, embedding=embedding)
            print(f"Loaded {len(splits)} chunks into Chroma vectorstore.")
            return vectorstore.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            raise RuntimeError(f"Error initializing Chroma vectorstore. Ensure `chromadb` is installed.\nError: {e}")

    # -------------------------------
    def create_chain(self, retriever):
        """Create a conversational retrieval chain with the new memory API."""
        if not retriever:
            raise ValueError("Retriever not provided to create_chain().")

        prompt = PromptTemplate.from_template("""
You are a friendly, knowledgeable AI tutor who explains things clearly.
Use the context carefully, include examples when helpful, and guide understanding.

If the context doesn’t fully answer the question, say:
"I’m not completely sure based on the provided context, but here’s what I can infer:"

---
**Context:**
{context}

**Conversation History:**
{chat_history}

**User’s Question:**
{question}

**Your Detailed, Example-Rich Answer:**
""")

        base_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        # Replace ConversationBufferMemory with new message history system
        store = {}

        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        return RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

