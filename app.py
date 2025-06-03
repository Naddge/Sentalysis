from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load models
try:
    ocr_pipeline = pipeline("image-to-text", model="nadchan/trocr-encoder-only")
    emotion_pipeline = pipeline("sentiment-analysis", model="nadchan/Sentalysis")
except Exception as e:
    print(f"Errore nel caricamento dei modelli: {e}")
    ocr_pipeline = None
    emotion_pipeline = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if "image" not in request.files:
        return render_template("index.html", error="Nessun file selezionato")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="Nessun file selezionato")

    if not ocr_pipeline or not emotion_pipeline:
        return render_template("index.html", error="Modelli non caricati correttamente")

    try:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        image = Image.open(image_path)

        # extract text
        ocr_result = ocr_pipeline(image)
        text = ocr_result[0]["generated_text"]

        # categorize
        emotion_result = emotion_pipeline(text)[0]

        label_map = {
            "LABEL_0": "Sadness",
            "LABEL_1": "Joy",
            "LABEL_2": "Love",
            "LABEL_3": "Anger",
            "LABEL_4": "Fear",
            "LABEL_5": "Surprise"
        }

        raw_label = emotion_result["label"]
        label = label_map.get(raw_label, "unknown")

        emotion = {
            "label": label,
            "score": emotion_result["score"]
        }

        return render_template(
            'index.html',
            text=text,
            emotion=emotion,
            image_url=url_for('static', filename=f'uploads/{filename}')
        )

    except Exception as e:
        return render_template("index.html", error=f"Errore durante l'elaborazione: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
