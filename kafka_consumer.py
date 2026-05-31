from kafka import KafkaConsumer, KafkaProducer
from transformers import pipeline
import json
import redis
import sys
import os
from database import SessionLocal, Prediction

sys.stdout.flush()

# Create consumer once
consumer = KafkaConsumer(
    'prediction_inputs', # topic to read from
    bootstrap_servers = 'kafka:9092',
    group_id = 'ml-workers', # consumer group name
    value_deserializer = lambda v: json.loads(v.decode('utf-8')),
    enable_auto_commit = False,
    auto_offset_reset='earliest'   
)

# Create producer once
producer = KafkaProducer(
        bootstrap_servers = 'kafka:9092',
        value_serializer = lambda v: json.dumps(v).encode('utf-8')
    )

redis_client  = redis.Redis(
    host = os.getenv('REDIS_HOST','localhost'),
    port = 6379,
    db = 2 # Separate db for cache (db=0) and Celery (db=1)
)


# Load model once
model = pipeline('sentiment-analysis', model = 'distilbert-base-uncased-finetuned-sst-2-english')

for message in consumer:
    data = message.value # dict sent from FastAPI
    print(data, flush=True) 
    result = model(data['text'])

    result_message = {
        'job_id' : data['job_id'],
        'text' : data['text'],
        'prediction' : result[0]['label'].lower(),
        'confidence': round(result[0]['score'],4)
    }

    producer.send('prediction_results',value = result_message)

    # db = SessionLocal()
    # record = Prediction(
    #     text=message['text'],
    #     prediction=message['prediction'],
    #     confidence=message['confidence']
    # )
    # db.add(record)
    # db.commit()
    # db.close()

    producer.flush()

    redis_client.set(data['job_id'],json.dumps(result_message),ex=86400)



    consumer.commit()

'''
message.value      → your actual data (the dict)
message.offset     → position in partition
message.partition  → which partition it came from
message.topic      → which topic
'''