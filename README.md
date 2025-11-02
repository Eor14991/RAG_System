# 🧠 AI Knowledge Assistant (RAG)

This is a Streamlit application that uses a Retrieval-Augmented Generation (RAG)
model to answer questions based on a provided knowledge base (PDF documents).
It's powered by LangChain and Google's Gemini models.

## Features

- Chat with your own PDF documents.
- Uses `FastEmbed` for local, fast embeddings.
- Uses `Chroma` as a local vector store.
- Maintains conversational history for follow-up questions.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Eor14991/RAG_System.git](https://github.com/Eor14991/RAG_System.git)
    cd RAG_System
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` and add your Google API key:
    ```
    GOOGLE_API_KEY="your-gemini-api-key-here"
    ```

5.  **Add your documents:**
    Place your PDF files inside the `documents/` directory.

## How to Run

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
