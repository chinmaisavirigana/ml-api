from fastapi import FastAPI,HTTPException
from database import SessionLocal, Prediction
from prometheus_fastapi_instrumentator import Instrumentator
import redis
from kafka_producer import send_prediction_request

import json
import os

app = FastAPI()

Instrumentator().instrument(app).expose(app)

# Connect to redis
import os

# Cache for predictions (db=0)
cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0
)

# Result store — kafka_consumer writes here, /result reads from here (db=2)
result_cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=2)


@app.get("/")
def home():
    return {"message": "Hello! My first API is alive."}

@app.get("/health")
def health():
    return {"status": "Running", "version": "1.0"}

@app.get("/predict")
def predict(text: str):
    
    #Step 1 - Check cache first
    cached = cache.get(text)
    if cached:
        # convert dict into string - Redis stores only string
        result = json.loads(cached)
        result['source'] = 'cache'
        return result

    job_id = send_prediction_request(text)

    return {
        'job_id' : job_id,
        'status' : 'processing',
        'message': 'Your prediction is being processed, Check /result/{job_id}'
    }

@app.get('/result/{job_id}')
def get_result(job_id:str):
    # db = SessionLocal()
    # record = db.query(Prediction).filter(Prediction.id==job_id).first()
    # db.close()

    cached = result_cache.get(job_id)
    
    if cached is None:
        return {'job_id': job_id, 'status':'processing'}
    
    result = json.loads(cached)
    return {
        'job_id' : job_id,
        "status" : 'complete',
        "result" : result
    }
    '''
    # To look up the result     
    job = run_prediction.AsyncResult(job_id)

    if job.state == 'PENDING':
        return {'job_id':job_id, 'status':'processing'}

    if job.state == 'FAILURE':
        return {'job_id':job_id, 'status':'failed'}

    if job.state == 'SUCCESS':
        result = job.result

        #save to cache now that we have the result
        cache.set(result['input'],json.dumps(result),ex=60)

        db = SessionLocal()
        record = Prediction(
            text = result['input'],
            prediction  = result['prediction'],
            confidence=result.get('confidence',0.0)
        )
        db.add(record)
        db.commit()
        db.close()
        return {'job_id':job_id, 'status':'complete','result':result}
        '''

    # # Step 2 - cache miss, run the model
    # positive_words = ["good", "great", "love", "amazing", "excellent"]
    # is_positive = any(word in text.lower() for word in positive_words)
    # result = "positive" if is_positive else "negative"
    # confidence = 0.87

    # # Save to database — this is new
    # db = SessionLocal()
    # record = Prediction(
    #     text=text,
    #     prediction=result,
    #     confidence=confidence
    # )
    # db.add(record)
    # db.commit()
    # db.close()

    # result =  {
    #     "input": text,
    #     "prediction": result,
    #     "confidence": confidence
    # }
    
    # # convert string into dict back
    # # Step 4 - save to cache for next time
    # cache.set(text, json.dumps(result), ex=60)  # expires in 60 seconds
    # result['source'] = 'model'
    # return result
   

  

@app.get("/history")
def history():
    # Read all past predictions from database
    db = SessionLocal()
    records = db.query(Prediction).all()
    db.close()

    return [
        {
            "id": r.id,
            "text": r.text,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "created_at": str(r.created_at)
        }
        for r in records
    ]

@app.get("/history/{id}")
def get_one(id:int):
    db = SessionLocal()
    record = db.query(Prediction).filter(Prediction.id==id).first()
    db.close()

    if record is None:
        raise HTTPException(status_code=404, detail='Prediction not found')
    return record

'''
/predict    → Kafka queue → async ML inference → instant response
/result     → Redis lookup → complete result when ready
/history    → SQLite → all past predictions
/health     → status check
/metrics    → Prometheus scraping
'''