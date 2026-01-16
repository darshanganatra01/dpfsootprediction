FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ src/
COPY run_api.py .

# copy model artifacts
COPY models/ models/

EXPOSE 8000

CMD ["python", "run_api.py"]
