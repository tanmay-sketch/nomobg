FROM python:3.10-slim

WORKDIR /app

# Copy only production requirements file first to leverage Docker cache
COPY requirements.prod.txt ./requirements.txt

# Install dependencies with cleanup in the same layer
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy application code
COPY . .

# Create model directories and set proper structure
RUN mkdir -p /app/u2net/model

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
# Print directory structure for debugging\n\
echo "Directory structure:"\n\
find /app -type d | sort\n\
echo "Looking for model at /app/u2net/model/u2netp.pth"\n\
ls -la /app/u2net/model/ || echo "Model directory is empty"\n\
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --log-level debug &\n\
sleep 5\n\
streamlit run app/frontend/home.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > start.sh && chmod +x start.sh

# Run both services
CMD ["./start.sh"]


