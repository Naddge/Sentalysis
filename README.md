# 🧠 AI Pipeline: Handwritten Text Recognition + Sentiment Analysis

This repository contains a two-stage AI system that:

1. ✍️ **Recognizes handwritten text** from images using an OCR model.
2. 💬 **Analyzes the sentiment** of the recognized text using a transformer-based sentiment classifier.

## 🧠 Models

### ✍️ Handwritten Text Recognition

* Converts images of handwritten text to machine-readable strings.
* Model used:
* Input: `.jpg` or `.png` images
* Output: Extracted plain text

### 💬 Sentiment Analysis

* Classifies text as **positive**, **neutral**, or **negative**.
* Model used: \[google-bert/bert-base-uncased]
* Input: Extracted text
* Output: Sentiment label

## 📊 Datasets Used

* **OCR:** [OCR Dataset](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/) 
* **Sentiment:** [Emotion Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) 

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Author & Contributions

Developed by **Nadia Ge**
Feel free to open issues or pull requests for suggestions, improvements, or bug reports.

