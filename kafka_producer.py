from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import uuid
import os
import time

_producer = None

def get_producer():
    global _producer
    if _producer is None:
        kafka_host = os.getenv('KAFKA_HOST','localhost')
        retries = 5
        for i in range(retries):
            try:
                _producer = KafkaProducer(
                    bootstrap_servers = f"{kafka_host}:9092",
                    value_serializer = lambda v: json.dumps(v).encode('utf-8')
                )
            except NoBrokersAvailable:
                print(f'Kafka not ready, retrying in 3s... ({i+1}/{retries})')
                time.sleep(3)

    return _producer

def send_prediction_request(text:str) -> str:
    job_id = str(uuid.uuid4())

    message = {
    'job_id' : job_id,
    'text' : text
    }

    get_producer().send('prediction_inputs', value = message)
    #  wait until all messages are actually sent to Kafka before moving on
    get_producer().flush()

    return job_id