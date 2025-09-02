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

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set tesseract.exe path here if on Windows
TESSERACT_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe" # Update this path if Tesseract is installed elsewhere

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

genai.configure(api_key=GEMINI_API_KEY)

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
    img = deskew(image_np)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morphed

def tesseract_ocr(image_np, psm=6, oem=3, lang="eng"): # Removed hin for broader compatibility initially
    config = f"--oem {oem} --psm {psm}"
    # Convert numpy array to PIL Image for pytesseract
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(image_pil, lang=lang, config=config)
    return text

def run_gemini_repair(image_pil, raw_text, prompt_mode="clean"):
    """
    prompt_mode:
      - "clean": produce clean reconstructed text using both inputs.
      - "extract": return structured fields (entities) as JSON.
    """
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

st.set_page_config(layout="wide", page_title="Image Summarization with Ollama VLM + OCR & Gemini")

st.title("Image Summarization with Ollama VLM + OCR & Gemini")

# Sidebar for model selection and instructions
st.sidebar.header("Configuration")
ollama_model = st.sidebar.text_input("Ollama Model Name", value="llava")
if not GEMINI_API_KEY:
    gemini_api_key_input = st.sidebar.text_input("Gemini API Key", type="password")
    if gemini_api_key_input:
        genai.configure(api_key=gemini_api_key_input)
        st.sidebar.success("Gemini API Key configured!")
    else:
        st.sidebar.warning("Please enter your Gemini API Key or set GEMINI_API_KEY in a .env file.")
else:
    st.sidebar.success("Gemini API Key loaded from environment variables.")

st.sidebar.markdown("""---
**Instructions:**
1.  Ensure Ollama server is running.
2.  Pull the selected VLM (e.g., `ollama pull llava`).
3.  Ensure Tesseract OCR is installed and its path is correctly configured in `app.py`.
4.  Upload an image using the browser.
5.  Click 'Generate Summary'.
""")

# Main content area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "gif"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

if st.button("Generate Summary"):
    if image is not None:
        if not GEMINI_API_KEY and not gemini_api_key_input:
            st.error("Please provide a Gemini API Key to proceed with text summarization.")
        else:
            with st.spinner("Generating image summary with Ollama and performing OCR with Tesseract & Gemini..."):
                try:
                    # Ollama VLM for image summarization
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    ollama_response = ollama.chat(model=ollama_model, messages=[
                        {
                            'role': 'user',
                            'content': 'Describe this image in detail.',
                            'images': [img_base64]
                        },
                    ])
                    st.subheader("Ollama Image Summary:")
                    st.write(ollama_response['message']['content'])

                    # Tesseract OCR and Gemini Text Repair/Extraction
                    # Convert PIL Image to OpenCV format (numpy array)
                    image_np = np.array(image)
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    preprocessed_image = preprocess_for_ocr(image_cv)
                    raw_text = tesseract_ocr(preprocessed_image)
                    st.subheader("Raw OCR Text:")
                    st.write(raw_text if raw_text else "No text found by OCR.")

                    if raw_text:
                        gemini_clean_text = run_gemini_repair(image, raw_text, prompt_mode="clean")
                        st.subheader("Gemini Refined Text:")
                        st.write(gemini_clean_text)

                        # Optional: Gemini for entity extraction
                        # gemini_entities_json_str = run_gemini_repair(image, raw_text, prompt_mode="extract")
                        # st.subheader("Gemini Extracted Entities (JSON):")
                        # st.json(json.loads(gemini_entities_json_str))

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.info("Please ensure Ollama server is running, model is pulled, Tesseract is installed and configured, and Gemini API key is valid.")
    else:
        st.warning("Please upload an image first.")
