# Start with a base image that has Python installed
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /code

RUN mkdir /data

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies specified in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .