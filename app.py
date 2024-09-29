import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

# Load model on CPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map=None
).to("cpu")  # Ensure the model is on CPU

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Streamlit app
st.title("OCR Application with Keyword Search")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare the image for the model
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
                    "image": img,  # Pass the image object directly
                },
                {
                    "type": "text",
                    "text": "Read and extract ALL text visible in this image. Provide ONLY the actual words, numbers, and characters you see, exactly as they appear."
                },
            ],
        }
    ]

    # Process the image for inference
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
    inputs = inputs.to("cpu")  # Send the inputs to CPU

    # Inference on CPU
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Display the extracted text
    extracted_text = output_text[0]
    st.subheader("Extracted Text")
    st.write(extracted_text)

    # Keyword Search
    keyword = st.text_input("Enter keyword to search in the extracted text")
    if keyword:
        if keyword.lower() in extracted_text.lower():
            highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
            st.subheader("Keyword Found")
            st.write(highlighted_text, unsafe_allow_html=True)
        else:
            st.write("Keyword not found in the extracted text.")
