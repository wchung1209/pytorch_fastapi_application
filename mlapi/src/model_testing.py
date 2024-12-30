from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

model_path = os.path.join(os.path.dirname(__file__), "../distilbert-base-uncased-finetuned-sst2")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
print(classifier(["I love FastAPI!", "I hate bugs."]))
