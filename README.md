# 📚 RAG-Powered Document Q&A System

This repository presents a practical implementation of a Retrieval Augmented Generation (RAG) system, designed to enable intelligent question-answering and semantic search over custom PDF documents. It demonstrates how to transform unstructured text into a searchable knowledge base, allowing an AI chatbot to provide accurate, context-aware responses.

## ✨ Features

* **PDF Document Ingestion:** Easily upload and process PDF files, extracting their content for indexing.
* **OpenAI Embeddings:** Leverages `text-embedding-3-small` from **OpenAI** to convert document chunks and user queries into high-quality vector embeddings.
* **Pinecone Vector Store:** Utilizes **Pinecone** as a scalable and efficient vector database for storing and managing the document embeddings.
* **Intelligent Chunking:** Employs `RecursiveCharacterTextSplitter` with `tiktoken` to intelligently split documents into semantically meaningful chunks, preserving context and optimizing retrieval.
* **Contextual Question Answering:** Integrates a **LangChain** `RetrievalQA` chain with **OpenAI's GPT-3.5 Turbo** to answer user questions by dynamically retrieving the most relevant document sections.
* **Semantic Search Functionality:** Provides a dedicated endpoint for performing semantic searches, returning relevant document passages along with their original page numbers.

## 🚀 Technologies Used

* **Python**
* **FastAPI:** For building the web API endpoints.
* **PyMuPDF (`fitz`):** For efficient PDF text extraction.
* **OpenAI Python Library:** For generating text embeddings and LLM responses.
* **Pinecone Python Client:** For interacting with the Pinecone vector database.
* **LangChain:** The powerful framework orchestrating the RAG pipeline, including text splitting, embeddings, vector store integration, and LLM chains.
* **Tiktoken:** For accurate token counting during text chunking.

## 💻 Installation & Setup

*(Add detailed instructions here for setting up the environment, installing dependencies, and configuring API keys.)*

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt # (You'll need to create a requirements.txt file)
    ```
    *(Alternatively, list individual packages: `pip install fastapi uvicorn python-dotenv PyMuPDF openai langchain-openai langchain-pinecone tiktoken`)*
4.  Set up environment variables:
    Create a `.env` file in the root directory with your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    PINECONE_API_KEY="your_pinecone_api_key_here"
    ```
5.  Place your `witcher.pdf` (or any other PDF you want to use) in the same directory as your main application file.

## 💡 Usage

*(Add instructions on how to run the FastAPI app and interact with the endpoints.)*

1.  Run the FastAPI application:
    ```bash
    uvicorn main:app --reload # Assuming your main FastAPI app is in main.py
    ```
2.  Access the API documentation at `http://127.0.0.1:8000/docs` (or your configured port).
3.  **Process your PDF:** Use the `/save-pdf` endpoint (POST request) to ingest your document and create embeddings in Pinecone.
4.  **Chat with the document:** Use the `/chat` endpoint (POST request) to ask questions about the document.
5.  **Perform semantic search:** Use the `/semantic_search_page` endpoint (POST request) to find relevant passages based on a query.

---

Feel free to explore the code and adapt it for your own RAG projects!
