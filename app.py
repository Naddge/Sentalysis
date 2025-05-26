from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import torch
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Modello OCR (TrOCR)
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Modello per riconoscimento emozioni
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# Etichette delle emozioni
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Nessuna immagine inviata"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # OCR
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Riconoscimento emozione
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    predicted_class = logits.argmax().item()
    emotion = labels[predicted_class]

    return jsonify({
        "text": text,
        "emotion": emotion
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import send_from_directory
import os

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')