import streamlit as st
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# Load the model and processor
@st.cache_resource  # Cache the model for faster loading
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    return model, processor

model, processor = load_model()

def extract_text(image):
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
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Read and extract ALL text visible in this image. Provide ONLY the actual words, numbers, and characters you see, exactly as they appear. If the text is in Hindi, give the output in Hindi characters. If the text is in English, give the output in English."
                },
            ],
        }
    ]

    # Prepare inputs
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference to generate text
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def highlight_keywords(text, keyword):
    # Case-insensitive highlighting of the keyword
    highlighted_text = text.replace(keyword, f"**{keyword}**")
    return highlighted_text

# Streamlit app layout
st.title("OCR Web Application")
st.write("Upload an image to extract text, and search for specific keywords.")

# File upload
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Extract text from image
    with st.spinner('Extracting text...'):
        extracted_text = extract_text(uploaded_image)

    # Display extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Keyword search and highlighting
    keyword = st.text_input("Enter keyword to highlight")
    if keyword:
        st.subheader("Text with Highlighted Keywords:")
        highlighted_text = highlight_keywords(extracted_text, keyword)
        st.markdown(highlighted_text)
