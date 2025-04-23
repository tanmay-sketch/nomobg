from fastapi import FastAPI, File, UploadFile
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

# Add U2NET directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        try:
            # Check if the file is a valid image
            with Image.open(temp_path) as test_img:
                # Force processing to check for corrupt images
                test_img.load()
                print(f"Input image mode: {test_img.mode}, size: {test_img.size}")
        except Exception as img_error:
            return {"error": f"Invalid input image: {str(img_error)}"}
            
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load model (U2NETP is smaller and faster)
        net = U2NETP(3, 1)
        model_path = os.path.join(u2net_dir, "model", "u2netp.pth")
        
        # Check if model exists
        if not os.path.exists(model_path):
            return {"error": f"Model file not found at: {model_path}"}
        
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

# Run the server with: uvicorn app.api:app --reload 