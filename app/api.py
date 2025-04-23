from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import tempfile
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import sys
import magic

# Add U2NET directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")
# Remove 'app' from path if we're in a Docker container
if current_dir.startswith('/app/app'):
    u2net_dir = os.path.join('/app', "u2net")
else:
    u2net_dir = os.path.join(current_dir, "u2net")
model_dir = os.path.join(u2net_dir, "model")
print(f"U2NET directory: {u2net_dir}")
print(f"Model directory: {model_dir}")

if u2net_dir not in sys.path:
    sys.path.append(u2net_dir)
if model_dir not in sys.path:
    sys.path.append(model_dir)

# Import U2NET models
try:
    from app.u2net.model.u2net import U2NET, U2NETP
    print("Successfully imported U2NET models")
except ImportError as e:
    print(f"Error importing U2NET models: {e}")
    # try:
    #     import sys
    #     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #     from app.u2net.model.u2net import U2NET, U2NETP
    #     print("Successfully imported U2NET models using alternative path")
    # except ImportError as e2:
    #     print(f"Error importing U2NET models with alternative path: {e2}")
    #     # Last resort, try direct import
    #     try:
    #         from u2net import U2NET, U2NETP
    #         print("Successfully imported U2NET models directly")
    #     except ImportError as e3:
    #         print(f"Failed all import attempts for U2NET models: {e3}")

# Try to import HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC support enabled")
except ImportError:
    print("pillow-heif not available - HEIC images may not work")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def normPRED(predicted_map):
    ma = np.max(predicted_map)
    mi = np.min(predicted_map)
    map_normalize = (predicted_map - mi) / (ma-mi)
    return map_normalize

def apply_mask(image, mask):
    # Convert mask to RGBA
    # Ensure image is a PIL Image object
    if isinstance(image, str):
        image = Image.open(image).convert('RGBA')
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image object or a file path")
    else:
        image = image.convert('RGBA')
        
    # Ensure mask is in correct format for PIL
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    
    # Convert mask to PIL Image and resize to match original image
    mask_img = Image.fromarray(mask).convert('L')
    mask_img = mask_img.resize(image.size, Image.BILINEAR)
    
    # Create a new RGBA image
    result = Image.new('RGBA', image.size, (0, 0, 0, 0))
    # Paste the original image
    result.paste(image, (0, 0))
    # Apply the mask as alpha channel
    result.putalpha(mask_img)
    
    return result

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    """
    Remove background from an image using U2NET model.
    """
    # Get file extension
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        try:
            # Open and convert to RGB if needed
            image = Image.open(temp_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                # Save as JPG for processing
                rgb_path = temp_path + ".jpg"
                image.save(rgb_path)
                temp_path = rgb_path
        except Exception as img_error:
            return {"error": f"Invalid input image: {str(img_error)}"}
            
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load model (U2NETP is smaller and faster)
        net = U2NETP(3, 1)
        
        # Try multiple possible paths for the model, with special handling for Docker
        possible_paths = [
            os.path.join(model_dir, "u2netp.pth"),  # Path from current script
            "/app/u2net/model/u2netp.pth",          # Direct Docker path
            "/app/app/u2net/model/u2netp.pth",      # Double app Docker path
        ]
        
        # Debug information
        for path in possible_paths:
            print(f"Checking model path: {path}, exists: {os.path.exists(path)}")
        
        # Find the first path that exists
        model_path = next((path for path in possible_paths if os.path.exists(path)), None)
        
        # If model not found, try to handle Docker path issues
        if not model_path:
            docker_model_path = "/app/u2net/model/u2netp.pth"
            docker_app_model_path = "/app/app/u2net/model/u2netp.pth"
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(docker_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(docker_app_model_path), exist_ok=True)
            
            # Check if we need to create a symlink between paths
            if os.path.exists(docker_model_path) and not os.path.exists(docker_app_model_path):
                try:
                    # Create symlink from docker path to double app path
                    os.symlink(docker_model_path, docker_app_model_path)
                    model_path = docker_app_model_path
                    print(f"Created symlink: {docker_model_path} -> {docker_app_model_path}")
                except Exception as e:
                    print(f"Failed to create symlink: {e}")
            elif os.path.exists(docker_app_model_path) and not os.path.exists(docker_model_path):
                try:
                    # Create symlink from double app path to docker path
                    os.symlink(docker_app_model_path, docker_model_path)
                    model_path = docker_model_path
                    print(f"Created symlink: {docker_app_model_path} -> {docker_model_path}")
                except Exception as e:
                    print(f"Failed to create symlink: {e}")
            
            # Try to find model path again
            model_path = next((path for path in possible_paths if os.path.exists(path)), None)
        
        # Check if model exists
        if not model_path:
            error_msg = f"Model file not found at any of these locations: {possible_paths}"
            print(error_msg)
            return {"error": error_msg}
        
        print(f"Using model at: {model_path}")
        
        # Load model weights
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        
        # Prepare image for model
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        resize_shape = (320, 320)
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        
        # Open and prepare image
        image = Image.open(temp_path).convert("RGB")
        image_resize = image.resize(resize_shape, resample=Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_tensor = transforms(image_resize).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            predictions = net(image_tensor)
            prediction = predictions[0] 
        
        try:
            pred = torch.squeeze(prediction.cpu(), dim=(0, 1)).numpy()
            pred = normPRED(pred)
            pred = (pred * 255).astype(np.uint8)
            
            print(f"Prediction shape: {pred.shape}, min: {np.min(pred)}, max: {np.max(pred)}")
            
            output_image = apply_mask(image, pred)
            
            print(f"Output image mode: {output_image.mode}, size: {output_image.size}")
            
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            try:
                # Test if the image can be opened to validate it
                test_img = Image.open(img_byte_arr)
                test_img.verify()  # Verify it's a valid image
                img_byte_arr.seek(0)  # Reset position after verification
                
                # Return the image
                return StreamingResponse(
                    img_byte_arr, 
                    media_type="image/png",
                    headers={"Content-Disposition": "attachment; filename=result.png"}
                )
            except Exception as img_error:
                return {"error": f"Failed to create valid image: {str(img_error)}"}
        except Exception as proc_error:
            return {"error": f"Error processing prediction: {str(proc_error)}"}
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Background Removal API"}

@app.post("/test-upload/")
async def test_upload(file: UploadFile = File(...), request: Request = None):
    """
    Test endpoint to diagnose upload issues
    """
    try:
        # Get request information
        headers = dict(request.headers) if request else {}
        
        # Get file info
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "headers": headers
        }
        
        # Read file content
        contents = await file.read()
        file_info["content_length"] = len(contents)
        
        # Try to identify file type
        try:
            mime = magic.Magic(mime=True)
            detected_type = mime.from_buffer(contents)
            file_info["detected_type"] = detected_type
        except ImportError:
            file_info["detected_type"] = "magic library not available"
        
        # Save to temp file and get image info
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        try:
            with Image.open(temp_path) as img:
                file_info["image_info"] = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size
                }
        except Exception as e:
            file_info["image_error"] = str(e)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return file_info
    except Exception as e:
        return {"error": str(e)}

# Run the server with: uvicorn app.api:app --reload 