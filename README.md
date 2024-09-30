
# OCR Application with Real-Time Token Streaming

This project is a Streamlit-based web application that uses the.Qwen2VL model to perform Optical Character Recognition (OCR) on uploaded images.
It features real-time token streaming and keyword search functionality, deployed on Hugging Face Spaces.

# Features

- Image upload and display
- Real-time OCR text extraction with token streaming
- Keyword search in extracted text
- Support for multiple languages (including Hindi and English)

# Prerequisites

- Python 3.7+
- pip (Python package manager)

# Installation

1. Clone this repository:
   ```
   git clone <your-repository-url>
   cd <your-project-directory>
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

# Running the Application Locally

1. Ensure you're in the project directory and your virtual environment is activated (if you're using one).

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

# Usage

1. Upload an image using the file uploader.
2. Wait for the OCR process to complete. You'll see the extracted text appear in real-time.
3. Use the keyword search feature to find specific words in the extracted text.

# Deployment on Hugging Face Spaces

This application is deployed on Hugging Face Spaces using the Streamlit SDK. Here's an overview of the deployment process:

1. Create a Hugging Face account: If you haven't already, sign up at [huggingface.co](https://huggingface.co/).

2. Create a new Space:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click on "Create new Space".
   - Choose "Streamlit" as the SDK.
   - Set up your Space with a name and visibility settings.

3. Prepare your repository:
   - Ensure your `app.py` and `requirements.txt` are in the root of your repository.
   - Add a `README.md` file (this file) to your repository.

4. Configure the Space:
   - In your Space's settings, under "Repository", link your GitHub repository.
   - Set the Python version if necessary.
   - Add any required secrets or environment variables.

5. Deploy:
   - Hugging Face Spaces will automatically deploy your app when you push changes to your linked repository.
   - You can also manually trigger a rebuild from the Space's settings.

6. Access your deployed app:
   - Your app will be available at `https://huggingface.co/spaces/<your-username>/<your-space-name>`.

Remember to update your `requirements.txt` file if you make any changes to your project dependencies.

# Notes

- The application uses CPU for inference by default. If you have a CUDA-capable GPU available on your deployment platform, you can modify the `device_map` and `to()` calls in `app.py` to use GPU acceleration.
- The model and processor are cached using Streamlit's `@st.cache_resource` decorator to improve performance on subsequent runs.

#DEMO IMAGE
![Ocr Model - a Hugging Face Space by ayush2607](https://github.com/user-attachments/assets/95c23f5d-e2b4-4583-acf7-8c494cb5cd0b)




Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

