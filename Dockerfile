FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (layer cache friendliness)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port HuggingFace Spaces expects
EXPOSE 7860

# Start FastAPI via uvicorn, binding to all interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]