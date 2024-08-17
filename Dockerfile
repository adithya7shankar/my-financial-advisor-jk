FROM ubuntu:latest

WORKDIR /app

ENV TZ=US/Pacific \
    DEBIAN_FRONTEND=noninteractive

ARG GITUN="Adithya Shankar"
ARG GITEMAIL="adithya7shankar@gmail.com"

# Install Python, pip, and git
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git

# Create and activate virtual environment
RUN python3 -m venv venv

ENV PATH="/app/venv/bin:$PATH"

# Copy the application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure git
RUN git config --global user.name "$GITUN" && \
    git config --global user.email "$GITEMAIL" && \
    git config --global init.defaultBranch main

# Expose port 80
EXPOSE 80

# Run the application
CMD ["python3", "main.py"]

