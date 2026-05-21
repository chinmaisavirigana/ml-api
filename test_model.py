from transformers import pipeline

# Load a pre-trained sentiment model

model = pipeline('sentiment-analysis')

tests = [
    "I love this product",
    "This is terrible",
    "not good at all",
    "I'm obsessed with this",
    "It's fine I guess"
    ]

for text in tests:
    result = model(text)
    print(f' Text: {text}')
    print(f' Result: {result}')
    print()