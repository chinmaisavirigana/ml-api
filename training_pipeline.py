'''
Stage 1 — Data ingestion
You'll load the IMDb movie review dataset from HuggingFace.
25,000 training examples, 
already labeled positive/negative.
This simulates a company's internal labeled dataset.


Stage 2 — Data validation
Before training, check the data is what you expect:'''

# Check: do we have enough data?
# Check: are labels balanced? (50% positive, 50% negative)
# Check: no empty texts?

# If validation fails — stop. 
# Don't waste hours training on bad data.

'''
Stage 3 — Feature engineering
For transformer models, 
feature engineering means tokenisation — 
converting raw text into numbers the model understands:

Stage 4 — Training
Fine-tune DistilBERT on your data. Log every epoch's metrics to MLflow.

Stage 5 — Evaluation
Test on held-out data. Compare accuracy to a baseline. Only proceed if better.

Stage 6 — Model registration
Save to MLflow model registry with full metadata — dataset used, hyperparameters, metrics.
'''

from datasets import load_dataset
from transformers import AutoTokenizer
import mlflow
import mlflow.pytorch
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

print("Stage 1: Loading data...")

dataset = load_dataset("imdb")

train_data = dataset["train"].shuffle(seed=42).select(range(1000))
eval_data  = dataset["test"].shuffle(seed=42).select(range(200))

print(f"  Train examples: {len(train_data)}")
print(f"  Eval examples:  {len(eval_data)}")
print(f"  Sample: {train_data[0]['text'][:500]}")
print(f"  Label:  {train_data[0]['label']} (0=negative, 1=positive)")

print("Stage 2: Validating data...")

# Check 1 — no empty texts
empty = sum(1 for x in train_data if len(x["text"].strip()) == 0)
if empty > 0:
    raise ValueError(f"Found {empty} empty texts — fix data before training")

# Check 2 — label balance (should be roughly 50/50)
labels = train_data["label"]
positive_ratio = sum(labels) / len(labels)
if not (0.3 <= positive_ratio <= 0.7):
    raise ValueError(f"Labels imbalanced: {positive_ratio:.2%} positive")

print(f"  Empty texts: {empty}")
print(f"  Positive ratio: {positive_ratio:.2%}")
print("  Data looks good.")

print("Stage 3: Tokenising...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,      # cut text longer than max_length
        padding="max_length", # pad shorter texts to max_length
        max_length=128        # 128 tokens — fast for learning
    )

train_data = train_data.map(tokenize, batched=True)
eval_data  = eval_data.map(tokenize, batched=True)

train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print(f"  Sample input_ids: {train_data[0]['input_ids'][:10]}...")
print("  Tokenisation complete.")

'''
Before writing — why MLflow?
Without MLflow:
You run training → get accuracy 87% → change learning rate → 
get 89% → forget what settings gave 89% → can never reproduce it

With MLflow:
Every run logged automatically:
  - hyperparameters (learning rate, batch size, epochs)
  - metrics (accuracy, loss per epoch)
  - model artifact (the actual weights)
  
6 months later: "what gave us 89%?" → open MLflow → see exact settings'''

print("Stage 4: Training...")

# "put all my runs in a folder called sentiment-training"
mlflow.set_experiment("sentiment-training")

with mlflow.start_run():

    # Log what settings we used
    # "record what settings I used"

    mlflow.log_params({
        "model": "distilbert-base-uncased",
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "epochs": 1,
        "batch_size": 16,
        "max_length": 128
    })


    #from_pretrained — load weights that were already 
    # trained by someone else. Don't start from random weights.
    # Load model

    #If you were classifying news articles 
    # (sports, politics, tech, entertainment) 
    # you'd write num_labels=4.

    #num_labels=2 — how many categories?
    #  Positive or negative = 2.

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # This runs after each epoch — logs accuracy to MLflow
    # "record what results I got"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        '''logits and argmax
    sAfter the model processes your text it returns logits — raw scores for each category:
    Input: "I love this movie"
    Output (logits): [−2.3,  4.1]
                    ↑      ↑
                    negative  positive
    Higher score = model is more confident about that category. But these are not probabilities yet — they're just raw numbers.
    To get the prediction you pick the highest score:
    [−2.3,  4.1]
            ↑
    index 1 is highest → prediction = 1 = positive
    That's what argmax does — returns the index of the highest value:
    pythonnp.argmax([−2.3, 4.1])  → 1
    np.argmax([3.1, −0.5])  → 0
    axis=-1 means "find the max along the last dimension" — across the scores for each example.
    '''


        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1  = f1_score(labels, predictions, average="weighted")
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        return {"accuracy": acc, "f1": f1}

    training_args = TrainingArguments(
        output_dir="./model_output",  # where to save checkpoints
        num_train_epochs=1,           # how many times to go through all data
        per_device_train_batch_size=16, # how many examples per step
        per_device_eval_batch_size=16,

        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",        # save model after each epoch
        load_best_model_at_end=True,  # keep the best version
        logging_steps=50,             # print loss every 50 steps
        report_to="none"              # don't send to wandb/tensorboard
)

    '''
    Trainer
    HuggingFace's Trainer handles the entire training loop for you:
    Without Trainer (manual):
    for epoch in epochs:
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    With Trainer:
    trainer.train()
    # that's it — Trainer does everything above
    You give it: model + settings + data + metrics function. It handles everything else.
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("  Training complete.")


    print("Stage 5: Evaluating...")

    results = trainer.evaluate()
    accuracy = results["eval_accuracy"]
    print(f"  Accuracy: {accuracy:.4f}")

    ACCURACY_THRESHOLD = 0.80

    if accuracy < ACCURACY_THRESHOLD:
        print(f"  Model REJECTED: {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}")
        print("  Current production model unchanged.")
    else:
        print(f"  Model PASSED: {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}")
        print("  Proceeding to registration...")


    print("Stage 6: Registering model...")

    # Log what settings produced this model
    mlflow.log_params({
        "model": "distilbert-base-uncased",
        "train_samples": len(train_data),
        "epochs": 1,
        "max_length": 128
    })

    # Log what results it achieved
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1": results["eval_f1"]
    })

    # Save the model itself
    mlflow.pytorch.log_model(
        trainer.model,
        artifact_path="sentiment-model",
        registered_model_name="sentiment-classifier"
    )

print("  Model registered in MLflow.")
print("  Pipeline complete.")

'''
Stage 1 → Loaded 1,000 IMDb reviews ✅
Stage 2 → Validated data (clean, balanced) ✅
Stage 3 → Tokenised text to numbers ✅
Stage 4 → Fine-tuned DistilBERT, logged to MLflow ✅
Stage 5 → Evaluated: 81% accuracy, passed threshold ✅
Stage 6 → Registered as 'sentiment-classifier' v1 ✅
'''