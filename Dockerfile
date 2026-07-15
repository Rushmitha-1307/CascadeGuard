# Start with a standard environment that knows how to use your GPU
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Create a folder called 'app' inside the digital backpack
WORKDIR /app

# Copy your ingredient list and install them
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your code into the backpack
COPY main.py .

# Tell the computer to run your code whenever the backpack is opened
CMD ["python", "main.py"]