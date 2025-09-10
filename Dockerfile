FROM python:3.11.13

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get autoclean; apt-get update --allow-insecure-repositories; \
    apt-get install ffmpeg libsm6 libxext6 git gfortran libopenblas-dev \
    liblapack-dev zip -y && apt-get install build-essential libssl-dev libffi-dev gnupg -y \
    && apt-get clean

COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip

RUN pip3 install -r /requirements.txt
RUN pip3 install pip-system-certs --use-feature=truststore
RUN apt-get install --reinstall ca-certificates
RUN pip3 install --upgrade certifi
RUN pip3 install pyopenssl==24.2.1

RUN printf "openssl_conf = openssl_init\n\
[openssl_init]\n\
ssl_conf = ssl_sect\n\
[ssl_sect]\n\
system_default = system_default_sect\n\
[system_default_sect]\n\
Options = UnsafeLegacyRenegotiation" > /openssl.cnf

ENV OPENSSL_CONF=/openssl.cnf

COPY . /code

WORKDIR /code

EXPOSE 8000

CMD [ "uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000" ]