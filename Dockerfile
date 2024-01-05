ARG EXTRA_DEPS=base

FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
COPY . .rs-graph-lib-source/

# Install git and tree
RUN apt-get -y update && apt-get -y install \
    git \
    software-properties-common

# Install dependencies
RUN pip install --no-cache-dir .rs-graph-lib-source[${EXTRA_DEPS}]