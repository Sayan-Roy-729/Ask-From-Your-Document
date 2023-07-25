# Use the official Python 3.10 image based on Alpine Linux as the base image
FROM python:3.10

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the content of the current directory (the application files) into the /app directory of the container
COPY . /app

# Install the Python dependencies listed in the requirements.txt file using pip
RUN pip install -r requirements.txt

# Set the OpenAI API key environment variable
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set the HuggingFace API key environment variable
ENV HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN

# Set the PORT environmental variable
ENV PORT=$PORT

# Expose the port specified by the $PORT environment variable (this will need to be set at runtime)
EXPOSE $PORT

# Set the default command to run when a container based on this image is started
CMD python ./main.py
