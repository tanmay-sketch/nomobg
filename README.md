# NoMoBG 🎨

A simple background removal tool that uses U2NET to automatically remove backgrounds from images.

## Features

- Easy-to-use web interface
- Fast background removal using deep learning
- Transparent PNG output
- No sign-up required

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nomobg.git
   cd nomobg
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the U2NET model files:

   ```bash
   mkdir -p app/u2net/model
   # Download u2net.pth and u2netp.pth model files to app/u2net/model/
   ```

## Usage

### Run with Python

Start the application locally:

```bash
python -m uvicorn app.api:app --reload
```

In a separate terminal, start the frontend:

```bash
streamlit run app/frontend/home.py
```

Then open your browser to <http://localhost:8501>

### Run with Docker

Build and run the Docker container:

```bash
docker compose build
docker compose up -d
```

Then open your browser to <http://localhost:8501>

## How It Works

1. Upload any image through the web interface
2. The backend processes the image using the U2NET model
3. A transparent PNG with the background removed is returned
4. Download the processed image

## Project Structure

- `app/api.py`: FastAPI backend for image processing
- `app/frontend/home.py`: Streamlit frontend
- `app/u2net/`: U2NET model files and image processing logic
- `Dockerfile`: Container configuration for deployment
