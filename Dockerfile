# 1. Use a lightweight, official Python runtime as a parent image
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set up system-level environment flags for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Install basic system dependencies needed for compiling certain wheel libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy the requirements list first to leverage Docker's caching system
COPY requirements.txt .

# 7. Upgrade pip and install the application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Copy the remaining application files and directories into the container
COPY app.py .
COPY chroma_db/ ./chroma_db/

# 9. Expose the port that Gradio uses internally
EXPOSE 7860

# 10. Define the execution command to launch your dynamic application script
CMD ["python", "app.py"]
