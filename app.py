import streamlit as st
import ollama
from PIL import Image
import io
import base64

st.set_page_config(layout="wide", page_title="Image Summarization with Ollama VLM")

st.title("Image Summarization with Ollama VLM")

# Sidebar for model selection and instructions
st.sidebar.header("Configuration")
model = st.sidebar.text_input("Ollama Model Name", value="llava")
st.sidebar.markdown("---\n**Instructions:**\n1.  Ensure Ollama server is running.\n2.  Pull the selected VLM (e.g., `ollama pull llava`).\n3.  Upload an image using the browser.\n4.  Click 'Generate Summary'.")

# Main content area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

if st.button("Generate Summary"):
    if image is not None:
        with st.spinner("Generating summary..."):
            try:
                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Call Ollama VLM
                response = ollama.chat(model=model, messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in detail.',
                        'images': [img_base64]
                    },
                ])
                st.subheader("Summary:")
                st.write(response['message']['content'])
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                st.info("Please ensure the Ollama server is running and the model is pulled.")
    else:
        st.warning("Please upload an image first.")
