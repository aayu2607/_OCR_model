import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from threading import Thread

@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map=None
    ).to("cpu")
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor

def process_file_streaming(img, model, processor):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specialized in reading and extracting text from images. Your task is to report the actual words and characters visible in the image, exactly as they appear, maintaining the original language (Hindi or English)."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": "Read and extract ALL text visible in this image. Provide ONLY the actual words, numbers, and characters you see, exactly as they appear."
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    # Stream tokens
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=600)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer

# Load model and processor once
model, processor = load_model()

# Streamlit app
st.title("OCR Application with Real-Time Token Streaming")

# Initialize session state variables
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Check if the uploaded image is different from the current one
    if st.session_state.current_image != uploaded_file:
        st.session_state.current_image = uploaded_file
        
        # Process the image with streaming
        streamer = process_file_streaming(img, model, processor)
        
        # Display streaming results
        st.subheader("Extracted Text (Streaming)")
        text_placeholder = st.empty()
        collected_text = ""
        
        for new_text in streamer:
            collected_text += new_text
            text_placeholder.markdown(collected_text)
        
        # Store the final extracted text
        st.session_state.extracted_text = collected_text

    else:
        # Display the previously extracted text
        st.subheader("Extracted Text")
        st.write(st.session_state.extracted_text)

# Keyword Search
keyword = st.text_input("Enter keyword to search in the extracted text")
if keyword and st.session_state.extracted_text:
    if keyword.lower() in st.session_state.extracted_text.lower():
        highlighted_text = st.session_state.extracted_text.replace(keyword, f"**{keyword}**")
        st.subheader("Keyword Found")
        st.markdown(highlighted_text)
    else:
        st.write("Keyword not found in the extracted text.")
elif keyword:
    st.write("Please upload an image first before searching.")
