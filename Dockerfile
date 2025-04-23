FROM python:3.10-slim

WORKDIR /app

# Set custom temp and pip cache directories to avoid filling up /tmp
ENV TMPDIR=/app/tmp \
    PIP_CACHE_DIR=/app/pip_cache

# Copy only production requirements to use Docker cache
COPY requirements.prod.txt ./requirements.txt

# Install dependencies and clean up
RUN mkdir -p $TMPDIR $PIP_CACHE_DIR && \
    pip install --no-cache-dir --cache-dir=$PIP_CACHE_DIR -r requirements.txt && \
    rm -rf $PIP_CACHE_DIR $TMPDIR /root/.cache/pip

# Copy rest of the app
COPY . .

# Create necessary directories and symlinks to handle double paths
RUN mkdir -p /app/u2net/model && \
    mkdir -p /app/app/u2net/model && \
    ln -sf /app/u2net/model/u2netp.pth /app/app/u2net/model/u2netp.pth

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Startup script to run both backend and frontend
RUN echo '#!/bin/bash\n\
# Debug directory structure\n\
echo "Directory structure:"\n\
find /app -type d | grep u2net\n\
echo "Symlinks:"\n\
ls -la /app/app/u2net/model/\n\
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &\n\
sleep 5\n\
streamlit run app/frontend/home.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > start.sh && chmod +x start.sh

CMD ["./start.sh"]


