# Usa un'immagine base con Python
FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY app.py .
COPY static/ ./static/
COPY templates/ ./templates/

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta 5000 (porta predefinita di Flask)
EXPOSE 5000

# Comando per avviare l'applicazione
CMD ["python", "app.py"]