

!pip install --upgrade transformers

!pip install transformers datasets torchvision

!pip install --upgrade torch torchvision torchaudio transformers datasets

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ssarkar445/handwriting-recognitionocr")

print("Path to dataset files:", path)

import pandas as pd

df = pd.read_csv("/kaggle/input/handwriting-recognitionocr/CSV/written_name_train.csv")
print(df.columns)

import pandas as pd

df = pd.read_csv("/kaggle/input/handwriting-recognitionocr/CSV/written_name_train.csv")
print(df.columns)
print(df.head())

from datasets import Dataset, Features, Value, Image
import pandas as pd

def prepare_dataset(csv_path, images_folder):
    df = pd.read_csv(csv_path)
    df["image"] = df["FILENAME"].apply(lambda x: f"{images_folder}/{x}")
    df = df[["image", "IDENTITY"]]
    features = Features({"image": Image(), "IDENTITY": Value("string")})
    return Dataset.from_pandas(df, features=features)

train_dataset = prepare_dataset("/kaggle/input/handwriting-recognitionocr/CSV/written_name_train.csv", "/kaggle/input/handwriting-recognitionocr/train_v2/train")
val_dataset = prepare_dataset("/kaggle/input/handwriting-recognitionocr/CSV/written_name_validation.csv", "/kaggle/input/handwriting-recognitionocr/validation_v2/validation")
test_dataset = prepare_dataset("/kaggle/input/handwriting-recognitionocr/CSV/written_name_test.csv", "/kaggle/input/handwriting-recognitionocr/test_v2/test")

print(train_dataset)
print(val_dataset)
print(test_dataset)

print(train_dataset[0])

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess(example):
    encoding = processor(
        images=example["image"],
        text_target=example["IDENTITY"],  
        padding="max_length",
        truncation=True,
        max_length=16,  
        return_tensors="pt"
    )
    return {
        "pixel_values": encoding["pixel_values"][0],
        "labels": encoding["labels"][0]
    }

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    num_train_epochs=3,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to="none"
)

from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import evaluate

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    preds_flat = [p for pred in decoded_preds for p in pred.split()]
    labels_flat = [l for label in decoded_labels for l in label.split()]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average='weighted', zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer_metric.compute(predictions=decoded_preds, references=decoded_labels),
        "wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels),
    }

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate(test_dataset)
print(metrics)
