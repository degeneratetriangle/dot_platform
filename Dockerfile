# Use an official Python runtime as a parent image
FROM python:2.7

# Set the working directory to /app
WORKDIR /dot_platform

# Copy the current directory contents into the container at /app
ADD . /dot_platform

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]