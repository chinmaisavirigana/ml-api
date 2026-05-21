from celery import Celery
import time
from transformers import pipeline

# Connect celery to Redis (Redis stores the jobs)

celery_app = Celery("worker",broker = 'redis://redis:6379/0', # where jobs come in 
                    backend = 'redis://redis:6379/1'  # where results go 
                )

# Load model once when worker starts
# NOT inside the function — loading takes 2 seconds
# You only want to pay that cost once

model = pipeline('sentiment-analysis', model = 'distilbert-base-uncased-finetuned-sst-2-english')

@celery_app.task

def run_prediction(text:str):
    # Simulate a slow model (like a real LLM)
    # time.sleep(3)

    # # Model logic
    # positive_words = ['good','love','great','amazing','excellent']
    # is_positive = any(word in text.lower() for word in positive_words)
    # prediction = 'positive' if is_positive else 'negative'

    # return {
    #     'input' : text,
    #     'prediction' : prediction,
    #     'confidence' : 0.87
    # }

    output = model(text)
    label = output[0]['label'].lower()
    score = round(output[0]['score'],4)

    return {
        'input' : text,
        'prediction' : label,
        'confidence' : score
    }