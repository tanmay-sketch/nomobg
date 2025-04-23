FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Run both services
CMD ["./start.sh"] 