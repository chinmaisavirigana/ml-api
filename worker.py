from celery import Celery
import time
from transformers import pipeline
import os

# Connect celery to Redis (Redis stores the jobs)

redis_host = os.getenv("REDIS_HOST", "localhost")

celery_app = Celery(
    "worker",
    broker=f"redis://{redis_host}:6379/0",
    backend=f"redis://{redis_host}:6379/1"
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