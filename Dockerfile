FROM python:3.10-slim

WORKDIR /code

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository content
COPY . .

# Expose port 7860 (default port for Hugging Face Spaces)
EXPOSE 7860

# Launch FastAPI backend with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
