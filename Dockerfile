FROM python:3.10-slim

WORKDIR /app

# Create dedicated pip cache dir to avoid filling up /tmp
ENV TMPDIR=/app/tmp \
    PIP_CACHE_DIR=/app/pip_cache

# Copy only production requirements file first to leverage Docker cache
COPY requirements.prod.txt ./requirements.txt

# Install dependencies with custom pip temp and cache directory
RUN mkdir -p $TMPDIR $PIP_CACHE_DIR && \
    pip install --no-cache-dir --cache-dir=$PIP_CACHE_DIR --build=/app/build -r requirements.txt && \
    rm -rf $PIP_CACHE_DIR $TMPDIR /root/.cache/pip /app/build

# Copy application code
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 &\n\
sleep 5\n\
streamlit run app/frontend/home.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > start.sh && chmod +x start.sh

CMD ["./start.sh"]


