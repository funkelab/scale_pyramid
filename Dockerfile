FROM python:3.9.10-slim-bullseye

WORKDIR /scale_pyramid

COPY requirements.txt requirements.txt

RUN \
    pip install --no-cache-dir pip==22.0.2 wheel==0.37.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY scale_pyramid.py scale_pyramid.py

ENTRYPOINT [ "python", "scale_pyramid.py" ]
