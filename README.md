# 🧠 Sentalysis: Sentence and Sentiment Analysis

This repository contains a two-stage AI system that:

1. ✍️ **Recognizes handwritten text** from images using an OCR model.
2. 💬 **Analyzes the sentiment** of the recognized text using a transformer-based sentiment classifier.

## 🧠 Models

### ✍️ Handwritten Text Recognition

* Converts images of handwritten text to machine-readable strings.
* Model used: \[ssarkar445/handwriting-recognitionocr\]
* Input: `.jpg` or `.png` images
* Output: Extracted plain text

### 💬 Sentiment Analysis

* Classifies text as **positive**, **neutral**, or **negative**.
* Model used: \[google-bert/bert-base-uncased\]
* Input: Extracted text
* Output: Sentiment label

## 📊 Datasets Used

* **OCR:** [OCR Dataset](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/) 
* **Sentiment:** [Emotion Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) 

## 📄 License

This project is released under the MIT License.

