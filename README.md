# Image Summarization with Ollama VLM in Streamlit

This project demonstrates how to use Ollama with a Vision-Language Model (VLM) to summarize images within a Streamlit application.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/PDF_SummarizerOllama.git
   cd PDF_SummarizerOllama/Pdf_Summarizer_Using_Rag
   ```
2. **Install Tesseract OCR:**

   * **Windows:** Download and install the Tesseract OCR executable from [https://tesseract-ocr.github.io/tessdoc/Downloads.html](https://tesseract-ocr.github.io/tessdoc/Downloads.html). Make sure to note the installation path (e.g., `C:/Program Files/Tesseract-OCR/tesseract.exe`). You will need to set this path in the `app.py` file.
   * **macOS:** `brew install tesseract`
   * **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install tesseract-ocr`
3. **Install Ollama:**
   If you haven't already, download and install Ollama from the official website: [https://ollama.ai/download](https://ollama.ai/download)
4. **Pull a Vision-Language Model (VLM):**
   Once Ollama is installed, pull a VLM. For example, to use the `llava` model:

   ```bash
   ollama pull llava
   ```

   You can explore other available models on the Ollama website.
5. **Set up Google Gemini API Key:**

   * Go to the Google AI Studio: [https://aistudio.google.com/](https://aistudio.google.com/)
   * Create a new API key.
   * Create a `.env` file in the `Pdf_Summarizer_Using_Rag` directory with the following content (replace with your actual key):
     ```
     GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
     TESSERACT_PATH="C:/Program Files/Tesseract-OCR/tesseract.exe" # Optional: Only if Tesseract is not in system PATH
     ```

     Replace `"YOUR_GEMINI_API_KEY"` with your actual API key and `"C:/Program Files/Tesseract-OCR/tesseract.exe"` with your Tesseract installation path if you choose to set it via environment variable.
6. **Create a Python Virtual Environment (recommended):**

   ```bash
   python -m venv venv
   ```
7. **Activate the Virtual Environment:**

   * **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
8. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Application

1. **Ensure Ollama server is running.**
2. **Ensure Tesseract OCR is installed and its path is correctly configured.** You can set the `TESSERACT_PATH` in your `.env` file or provide it directly in the Streamlit sidebar. Also, ensure necessary Tesseract language packs are installed for the selected OCR Languages (e.g., `eng.traineddata`, `hin.traineddata`).
3. **Ensure your `GEMINI_API_KEY` is set** in the `.env` file or provided in the Streamlit sidebar.
4. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```
   This will open the application in your web browser. You can then upload an image and generate a summary using the selected Ollama VLM, and also see the OCR extracted text and Gemini-repaired text. You can configure Ollama model, Tesseract path, and OCR languages from the sidebar.
