from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from PIL import Image
import io
import os

app = Flask(__name__)

# Carica i modelli all'avvio (potrebbe richiedere qualche secondo)
try:
    # Modello OCR per estrazione testo da immagini
    ocr_pipeline = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")
    
    # Modello per analisi emozioni del testo
    emotion_pipeline = pipeline("text-classification", model="nadchan/Sentalysis")
except Exception as e:
    print(f"Errore nel caricamento dei modelli: {e}")
    ocr_pipeline = None
    emotion_pipeline = None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return render_template("index.html", error="Nessun file selezionato")
    
    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="Nessun file selezionato")
    
    if not ocr_pipeline or not emotion_pipeline:
        return render_template("index.html", error="Modelli non caricati correttamente")
    
    try:
        # Leggi l'immagine
        image = Image.open(io.BytesIO(file.read()))
        
        # Estrai il testo
        ocr_result = ocr_pipeline(image)
        text = ocr_result[0]["generated_text"]
        
        # Analizza l'emozione
        emotion_result = emotion_pipeline(text)[0]
        emotion = {
            "label": emotion_result["label"],
            "score": emotion_result["score"]
        }
        
        return render_template("index.html", text=text, emotion=emotion)
    
    except Exception as e:
        return render_template("index.html", error=f"Errore durante l'elaborazione: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)