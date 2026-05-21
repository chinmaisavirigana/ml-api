# Start from an official python image
# This is base - like a clean machine with Python installed
FROM python:3.11-slim

# Set the working directory
# All commands from here
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of code
COPY . . 

# What to run when container starts
CMD ["uvicorn", "main:app","--host","0.0.0.0","--port","8000"]