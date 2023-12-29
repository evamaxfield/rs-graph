FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install git and tree
RUN apt-get -y update && apt-get -y install \
    git \
    software-properties-common

# Install dependencies
RUN pip install --no-cache-dir .