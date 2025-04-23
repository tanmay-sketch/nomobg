FROM python:3.10-slim

WORKDIR /app

# Copy only production requirements first to use Docker layer cache
COPY requirements.prod.txt ./requirements.txt

# Install dependencies with custom paths to avoid filling /tmp
RUN mkdir -p /app/tmp /app/pip_cache && \
    pip install --no-cache-dir --cache-dir=/app/pip_cache -r requirements.txt && \
    rm -rf /app/pip_cache /app/tmp /root/.cache/pip

# Copy the rest of the application
COPY . .

# Create symlink to satisfy the app’s hardcoded path needs
RUN mkdir -p /app/u2net/model /app/app/u2net/model && \
    ln -sf /app/u2net/model/u2netp.pth /app/app/u2net/model/u2netp.pth

# Expose ports
EXPOSE 8000 8501

# Startup script: Start FastAPI, then Streamlit
RUN echo '#!/bin/bash\n\
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &\n\
sleep 10\n\
streamlit run app/frontend/home.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]