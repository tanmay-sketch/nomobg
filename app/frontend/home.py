import streamlit as st
import requests
from PIL import Image
import io
import json
import os

# Get API URL from environment variable or use a default
# For the FastAPI server, use the internal Docker network address
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')

# Increase Streamlit's max upload size
st.set_option('server.maxUploadSize', 100)

# Page configuration
st.set_page_config(
    page_title="NoMoBG - Background Remover",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🎨 NoMoBG - Background Remover")
    st.markdown("""
    Upload an image and get your image without background!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Remove Background"):
            with st.spinner("Removing background..."):
                try:
                    # Prepare the file for upload - create a proper file object
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                    
                    # Make request to backend
                    response = requests.post(
                        "http://localhost:8000/remove-background/",
                        files=files
                    )
                    
                    # Log content type for debugging
                    content_type = response.headers.get('content-type', 'unknown')
                    # st.write(f"Debug - Content type: {content_type}, Length: {len(response.content)}")
                    
                    # Check the content type first
                    if 'json' in content_type:
                        # This is a JSON response, likely an error
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', 'Unknown error')
                            st.error(f"API Error: {error_msg}")
                            # Also show the raw JSON for debugging
                            st.code(response.text, language='json')
                        except:
                            st.error(f"Could not parse JSON response: {response.text}")
                    elif 'image' in content_type and response.status_code == 200:
                        try:
                            # Save response content to a file for inspection
                            img_bytes = io.BytesIO(response.content)
                            img_bytes.seek(0)
                            
                            # Display result
                            result_image = Image.open(img_bytes)
                            st.image(result_image, caption="Background Removed", use_column_width=True)
                            
                            # Download button
                            img_byte_arr = io.BytesIO()
                            result_image.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            st.download_button(
                                label="Download Image",
                                data=img_byte_arr,
                                file_name="background_removed.png",
                                mime="image/png"
                            )
                        except Exception as img_error:
                            st.error(f"Error processing image: {str(img_error)}")
                    else:
                        st.error(f"Unexpected response format (content-type: {content_type})")
                        st.code(response.text[:1000], language='text')
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 