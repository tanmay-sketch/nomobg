import streamlit as st
import requests
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="NoMoBG - Background Remover",
    page_icon="🎨"
)

def main():
    # Simple header
    st.title("🎨 NoMoBG")
    st.subheader("Background Removal Tool")
    
    # Input section
    st.write("Upload an image to remove its background.")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Process section
    if uploaded_file is not None:
        try:
            # Display original image
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Process button - simple, no centering
            if st.button("Remove Background"):
                with st.spinner("Processing image..."):
                    # Prepare the file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                    
                    # Make request to backend
                    response = requests.post(
                        "http://localhost:8000/remove-background/",
                        files=files
                    )
                    
                    # Handle the response
                    if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                        # Display result image
                        st.subheader("Result")
                        img_bytes = io.BytesIO(response.content)
                        img_bytes.seek(0)
                        result_image = Image.open(img_bytes)
                        st.image(result_image, use_container_width=True)
                        
                        # Download button - simple, no centering
                        img_byte_arr = io.BytesIO()
                        result_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        st.download_button(
                            label="Download Transparent PNG",
                            data=img_byte_arr,
                            file_name="background_removed.png",
                            mime="image/png"
                        )
                    else:
                        # Error handling
                        try:
                            error_data = response.json()
                            st.error(f"Error: {error_data.get('error', 'Unknown error')}")
                        except:
                            st.error("Error processing the image. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("NoMoBG • Powered by U2NETP")

if __name__ == "__main__":
    main() 