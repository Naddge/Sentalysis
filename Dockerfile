FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY static/ ./static/
COPY templates/ ./templates/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]