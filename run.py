import subprocess
import sys
import os
import time

def run_fastapi():
    print("Starting FastAPI backend...")
    fastapi_process = subprocess.Popen(
        ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return fastapi_process

def run_streamlit():
    print("Starting Streamlit frontend...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app/frontend/home.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return streamlit_process

def main():
    # Start FastAPI
    fastapi_process = run_fastapi()
    
    # Give FastAPI time to start
    time.sleep(3)
    
    # Start Streamlit
    streamlit_process = run_streamlit()
    
    print("NoMoBG is running!")
    print("FastAPI backend: http://localhost:8000")
    print("Streamlit frontend: http://localhost:8501")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main() 