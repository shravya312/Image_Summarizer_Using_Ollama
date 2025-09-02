# Image Summarization with Ollama VLM in Streamlit

This project demonstrates how to use Ollama with a Vision-Language Model (VLM) to summarize images within a Streamlit application.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PDF_SummarizerOllama.git
    cd PDF_SummarizerOllama/Pdf_Summarizer_Using_Rag
    ```

2.  **Install Ollama:**
    If you haven't already, download and install Ollama from the official website: [https://ollama.ai/download](https://ollama.ai/download)

3.  **Pull a Vision-Language Model (VLM):**
    Once Ollama is installed, pull a VLM. For example, to use the `llava` model:
    ```bash
    ollama pull llava
    ```
    You can explore other available models on the Ollama website.

4.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```

5.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

6.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  **Ensure Ollama server is running.**

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser. You can then upload an image and generate a summary using the selected Ollama VLM.