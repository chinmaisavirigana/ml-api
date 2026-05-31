from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# These are your "documents" — in production this would be
# thousands of articles, product descriptions, or support tickets
documents = [
    "How to make a healthy salad with vegetables",
    "Best lightweight meals for summer",
    "Car engine repair and maintenance guide",
    "Low calorie breakfast ideas",
    "How to fix a flat tyre",
    "Quick and easy soup recipes",
    "Machine learning model training tips",
    "Mediterranean diet food list",
    "Python programming for beginners",
    "Grilled chicken with light sauce recipe"
]

print(f"Documents to index: {len(documents)}")

print('Loading embedding model...')
model = SentenceTransformer('all-MiniLM-L6-v2')

print('Converting documents to vectors...')
embeddings = model.encode(documents)

print(f"  Each document → vector of {embeddings.shape[1]} numbers")
print(f"  All documents → matrix of shape {embeddings.shape}")
print(f"  Sample vector (first 5 numbers): {embeddings[0][:5]}")


print("Building FAISS index...")

dimension = embeddings.shape[1]  # 384

# Create a flat index — simplest type, exact search
index = faiss.IndexFlatL2(dimension)

# Add all document vectors to the index
index.add(embeddings.astype(np.float32))

print(f"  Vectors in index: {index.ntotal}")

print("Searching...")

query = "how to fix my vehicle"

# Convert query to vector — same model, same process
query_vector = model.encode([query])

# Search — find 3 most similar documents
k = 3
distances, indices = index.search(query_vector.astype(np.float32), k)

print(f"\nQuery: '{query}'")
print(f"\nTop {k} results:")
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. {documents[idx]}")
    print(f"     Distance: {distance:.4f}")