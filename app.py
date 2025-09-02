import streamlit as st
import ollama
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import pytesseract
import google.generativeai as genai
import json # Added json import
import time # Added time import

load_dotenv()
RAW_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAW_TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# Default Tesseract path for Windows if not set in environment
default_tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
TESSERACT_CMD = RAW_TESSERACT_PATH if RAW_TESSERACT_PATH else default_tesseract_cmd

OCR_LANGS = "eng+hin" # Using both English and Hindi
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Defer genai.configure until a key is confirmed
# genai.configure(api_key=GEMINI_API_KEY)

def deskew(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return image_np
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image_np.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(image_np):
    # Add padding to the image before further processing
    border_size = 20 # pixels
    padded_image = cv2.copyMakeBorder(image_np, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    img = deskew(padded_image) # Apply deskew to the padded image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morphed

def tesseract_ocr(image_np, lang, psm=6, oem=3):
    config = f"--oem {oem} --psm {psm}"
    # Convert numpy array to PIL Image for pytesseract
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(image_pil, lang=lang, config=config)
    return text

def run_gemini_repair(api_key, image_pil, raw_text, prompt_mode="clean"):
    if not api_key:
        raise ValueError("Gemini API Key is not configured.")
    genai.configure(api_key=api_key) # Configure with the provided API key
    model = genai.GenerativeModel("gemini-2.5-flash")
    base_system_prompt = (
        "You are an expert OCR post-processor. "
        "You are given (1) an original document image and (2) noisy OCR text from Tesseract.\n"
        "Task: reconstruct the most accurate text you can. "
        "Preserve reading order, headers, tables (as Markdown), and line breaks. "
        "If handwriting is present, read from the image to fill missing/incorrect words. "
        "Do not invent content that is not visible in the image."
    )
    extract_system_prompt = (
        "Extract key fields from the document. Respond with strict JSON only. "
        "Use keys: 'names', 'ids', 'dates', 'addresses', 'emails', 'phones', 'stamps_or_seals'. "
        "When unknown, use null or empty arrays. Do not include any commentary."
    )
    user_prompt = extract_system_prompt if prompt_mode == "extract" else base_system_prompt
    resp = model.generate_content([
        {"text": user_prompt},
        image_pil,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()

def save_run(run_dir, settings, ollama_summary, raw_text, gemini_text, entities_json_str=None):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "ollama_summary.txt"), "w", encoding="utf-8") as f:
        f.write(ollama_summary or "")
    with open(os.path.join(run_dir, "raw_ocr.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(os.path.join(run_dir, "gemini_text.txt"), "w", encoding="utf-8") as f:
        f.write(gemini_text or "")
    if entities_json_str:
        with open(os.path.join(run_dir, "entities.json"), "w", encoding="utf-8") as f:
            f.write(entities_json_str)
    st.success(f"Run saved to: {os.path.abspath(run_dir)}")
    return os.path.abspath(run_dir)

st.set_page_config(layout="wide", page_title="Image Summarization with Ollama VLM + OCR & Gemini")

st.title("Image Summarization with Ollama VLM + OCR & Gemini")

# Sidebar for model selection and instructions
st.sidebar.header("Configuration")
ollama_model = st.sidebar.text_input("Ollama Model Name", value="llava")

current_gemini_api_key = RAW_GEMINI_API_KEY
if not current_gemini_api_key:
    gemini_api_key_input = st.sidebar.text_input("Gemini API Key", type="password")
    if gemini_api_key_input:
        current_gemini_api_key = gemini_api_key_input
        st.sidebar.success("Gemini API Key configured from input!")
    else:
        st.sidebar.warning("Please enter your Gemini API Key or set GEMINI_API_KEY in a .env file.")
else:
    st.sidebar.success("Gemini API Key loaded from environment variables.")

current_tesseract_path = TESSERACT_CMD
tesseract_path_input = st.sidebar.text_input("Tesseract Executable Path", value=current_tesseract_path)
if tesseract_path_input:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path_input
    st.sidebar.success("Tesseract Path configured!")
else:
    st.sidebar.warning("Please enter the Tesseract executable path or set TESSERACT_PATH in a .env file.")

current_ocr_langs = OCR_LANGS
ocr_langs_input = st.sidebar.text_input("OCR Languages (e.g., eng, eng+hin)", value=current_ocr_langs)
if ocr_langs_input:
    OCR_LANGS = ocr_langs_input
    st.sidebar.success(f"OCR Languages set to: {OCR_LANGS}")
else:
    st.sidebar.warning("Please enter OCR languages to use (e.g., eng, eng+hin).")

st.sidebar.markdown("""---
**Instructions:**
1.  Ensure Ollama server is running.
2.  Pull the selected VLM (e.g., `ollama pull llava`).
3.  Ensure Tesseract OCR is installed and its path is correctly configured in the 'Tesseract Executable Path' field or in a `.env` file.
4.  Ensure necessary Tesseract language packs are installed for the selected OCR Languages.
5.  Upload an image using the browser.
6.  Click 'Generate Summary'.
""")

# Main content area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "gif"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

if st.button("Generate Summary"):
    if image is not None:
        if not current_gemini_api_key:
            st.error("Please provide a Gemini API Key to proceed with text summarization.")
        elif not pytesseract.pytesseract.tesseract_cmd:
            st.error("Please configure the Tesseract Executable Path.")
        else:
            with st.spinner("Generating image summary with Ollama and performing OCR with Tesseract & Gemini..."):
                try:
                    # Tesseract OCR and Gemini Text Repair/Extraction
                    image_np = np.array(image)
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    preprocessed_image = preprocess_for_ocr(image_cv)
                    raw_text = tesseract_ocr(preprocessed_image, lang=OCR_LANGS)
                    st.subheader("Raw OCR Text:")
                    st.write(raw_text if raw_text else "No text found by OCR.")

                    gemini_clean_text = ""
                    if raw_text:
                        gemini_clean_text = run_gemini_repair(current_gemini_api_key, image, raw_text, prompt_mode="clean")
                        st.subheader("Gemini Refined Text:")
                        st.write(gemini_clean_text)

                    # Ollama VLM for image summarization, now incorporating Gemini's refined text
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    ollama_prompt_content = "Describe this image in detail." 
                    if gemini_clean_text:
                        ollama_prompt_content += f" Also, consider the following extracted and refined text from the image for your description: \n\n{gemini_clean_text}"

                    ollama_response = ollama.chat(model=ollama_model, messages=[
                        {
                            'role': 'user',
                            'content': ollama_prompt_content,
                            'images': [img_base64]
                        },
                    ])
                    ollama_summary_content = ollama_response['message']['content']
                    st.subheader("Ollama Image Summary (with Gemini Context):")
                    st.write(ollama_summary_content)

                    # Save the run
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    run_dir = os.path.join(RUNS_DIR, f"run-{ts}")
                    settings = {
                        "ollama_model": ollama_model,
                        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
                        "ocr_languages": OCR_LANGS,
                        "gemini_model": "gemini-2.5-flash",
                        "image_file": uploaded_file.name # Save the original filename
                    }
                    save_run(run_dir, settings, ollama_summary_content, raw_text, gemini_clean_text)

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.info("Please ensure Ollama server is running, model is pulled, Tesseract is installed and configured, and Gemini API key is valid.")
    else:
        st.warning("Please upload an image first.")
