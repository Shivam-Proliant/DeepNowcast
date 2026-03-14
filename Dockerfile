# Use official PyTorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src /app/src
COPY ./api /app/api
# COPY deepnowcast_v1.pth /app/

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
