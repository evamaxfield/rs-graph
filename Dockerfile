# Use the node image but install python
FROM python:3.11

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
WORKDIR ./rs-graph
COPY . .

# Install git and upgrade pip
RUN apt-get -y update \
    && apt-get -y install git \
    && pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -v .