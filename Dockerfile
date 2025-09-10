# FROM python:3.10.18

# ENV DEBIAN_FRONTEND=noninteractive
# ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# RUN apt-get autoclean; apt-get update --allow-insecure-repositories; \
#     apt-get install ffmpeg libsm6 libxext6 git gfortran libopenblas-dev \
#     liblapack-dev zip -y && apt-get install build-essential libssl-dev libffi-dev gnupg -y \
#     && apt-get clean

# COPY requirements.txt /requirements.txt

# RUN pip3 install --upgrade pip

# RUN pip3 install -r /requirements.txt
# RUN pip3 install pip-system-certs --use-feature=truststore
# RUN apt-get install --reinstall ca-certificates
# RUN pip3 install --upgrade certifi
# RUN pip3 install pyopenssl==24.2.1

# RUN printf "openssl_conf = openssl_init\n\
# [openssl_init]\n\
# ssl_conf = ssl_sect\n\
# [ssl_sect]\n\
# system_default = system_default_sect\n\
# [system_default_sect]\n\
# Options = UnsafeLegacyRenegotiation" > /openssl.cnf

# ENV OPENSSL_CONF=/openssl.cnf

# COPY . /code

# WORKDIR /code

# EXPOSE 8000

# CMD [ "uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000" ]







# ---- Build Stage ----
# Use a full image to get all the build tools
FROM python:3.10.18 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install build-time dependencies and clean up in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 git gfortran libopenblas-dev \
    liblapack-dev zip build-essential libssl-dev libffi-dev gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

# Install python packages into the virtual env, disabling the cache
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir pip-system-certs --use-feature=truststore
RUN pip install --no-cache-dir --upgrade certifi
RUN pip install --no-cache-dir pyopenssl==24.2.1

# ---- Final Stage ----
# Use a lightweight slim image for the final application
FROM python:3.10.18-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libopenblas0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the OpenSSL config and application code
COPY . /code
RUN printf "openssl_conf = openssl_init\n\
[openssl_init]\n\
ssl_conf = ssl_sect\n\
[ssl_sect]\n\
system_default = system_default_sect\n\
[system_default_sect]\n\
Options = UnsafeLegacyRenegotiation" > /openssl.cnf

ENV OPENSSL_CONF=/openssl.cnf

WORKDIR /code

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD [ "uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000" ]