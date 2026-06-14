import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Download movielens small dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

import urllib.request
import zipfile
import os

if not os.path.exists('ml-latest-small'):
    print('Downloading MovieLens dataset...')
    urllib.request.urlretrieve(url,'ml-latest-small.zip')

    with zipfile.ZipFile('ml-latest-small.zip','r') as z:
        z.extractall('.')
    print('Downloaded.')

ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(f'Ratings: {len(ratings)}')
print(f'Users: {ratings['userId'].nunique()}')
print(f'Movies: {ratings['movieId'].nunique()}')
print(ratings.head())

# Map user and movie IDs to sequential integers starting from 0

user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user2idx = {uid:id for id, uid in enumerate(user_ids)}
movie2idx = {mid:id for id, mid in enumerate(movie_ids)}

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['movie_idx'] = ratings['movieId'].map(movie2idx)

num_users = len(user_ids)
num_movies = len(movie_ids)

print(f"Num users:  {num_users}")
print(f"Num movies: {num_movies}")
print(ratings[['user_idx', 'movie_idx', 'rating']].head())

'''User Tower:
  Input: user_idx (integer)
  Embedding layer: converts integer → 32 numbers
  Output: 32-dimensional user vector

Item Tower:
  Input: movie_idx (integer)
  Embedding layer: converts integer → 32 numbers
  Output: 32-dimensional movie vector

Score:
  Dot product of user vector × movie vector
  High score = user will like this movie
  Low score = user won't like this movie'''

import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, mum_users, num_movies, embedding_dim = 32):
        super().__init__()
 
        # User tower — converts user_id to a vector
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
       
        # Item tower — converts movie_id to a vector
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
    
    def forward(self, user_idx, movie_idx):
        # Get embeddings for this user and movie
        user_vec  = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)

         
        # Dot product = predicted relevance score
        score = (user_vec * movie_vec).sum(dim=1)
        return score
    
# Create model
model = TwoTowerModel(num_users, num_movies, embedding_dim=32)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)


from torch.utils.data import DataLoader, TensorDataset

# Split into train and test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convert to tensors
train_users  = torch.tensor(train_data['user_idx'].values, dtype=torch.long)
train_movies = torch.tensor(train_data['movie_idx'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)

test_users  = torch.tensor(test_data['user_idx'].values, dtype=torch.long)
test_movies = torch.tensor(test_data['movie_idx'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)

# DataLoader — batches the data
train_dataset = TensorDataset(train_users, train_movies, train_ratings)
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("Training...")
for epoch in range(5):
    model.train()
    total_loss = 0
    
    for user_batch, movie_batch, rating_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(user_batch, movie_batch)
        loss = criterion(predictions, rating_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/5 — Loss: {avg_loss:.4f}")

print("Training complete.")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_users, test_movies)
    test_loss = criterion(test_predictions, test_ratings)
    rmse = torch.sqrt(test_loss)
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"(Average rating prediction error: {rmse:.2f} stars)")

# Generate recommendations for a user
def recommend(user_id, top_k=10):
    model.eval()
    with torch.no_grad():
        user_idx = user2idx[user_id]
        
        # Score ALL movies for this user
        user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
        movie_tensor = torch.tensor(list(range(num_movies)), dtype=torch.long)
        
        scores = model(user_tensor, movie_tensor)
        
        # Get top K movies
        top_indices = scores.argsort(descending=True)[:top_k]
        
        # Convert back to original movie IDs
        idx2movie = {v: k for k, v in movie2idx.items()}
        top_movie_ids = [idx2movie[idx.item()] for idx in top_indices]
        
        return top_movie_ids

# Get recommendations for user 1
user_id = 1
recommendations = recommend(user_id, top_k=5)

# Load movie titles
movies = pd.read_csv("ml-latest-small/movies.csv")
print(f"\nTop 5 recommendations for User {user_id}:")
for movie_id in recommendations:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"  {title}")

import faiss

print("\nBuilding FAISS index for movie retrieval...")

# Step 1 — get ALL movie embeddings from item tower
model.eval()
with torch.no_grad():
    all_movie_indices = torch.tensor(list(range(num_movies)), dtype=torch.long)
    all_movie_embeddings = model.movie_embedding(all_movie_indices).numpy()

print(f"  Movie embeddings shape: {all_movie_embeddings.shape}")

# Step 2 — build FAISS index
dimension = 32
index = faiss.IndexFlatL2(dimension)
index.add(all_movie_embeddings.astype(np.float32))
print(f"  Vectors in index: {index.ntotal}")

# Step 3 — retrieve top candidates using FAISS
def retrieve_candidates(user_id, top_k=20):
    model.eval()
    with torch.no_grad():
        user_idx = user2idx[user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        user_embedding = model.user_embedding(user_tensor).numpy()

    # FAISS search — find top_k closest movie vectors
    distances, indices = index.search(user_embedding.astype(np.float32), top_k)

    # Convert indices back to movie IDs
    idx2movie = {v: k for k, v in movie2idx.items()}
    candidate_movie_ids = [idx2movie[idx] for idx in indices[0]]

    return candidate_movie_ids, distances[0]

# Test retrieval for user 1
candidates, distances = retrieve_candidates(user_id=1, top_k=10)

print(f"\nFAISS retrieved top 10 candidates for User 1:")
for movie_id, dist in zip(candidates, distances):
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"  {title} (distance: {dist:.4f})")

def rank_candidates(user_id, candidate_movie_ids):
    model.eval()
    with torch.no_grad():
        user_idx = user2idx[user_id]
        
        user_tensor = torch.tensor(
            [user_idx] * len(candidate_movie_ids), 
            dtype=torch.long
        )
        movie_tensors = torch.tensor(
            [movie2idx[mid] for mid in candidate_movie_ids],
            dtype=torch.long
        )
        
        scores = model(user_tensor, movie_tensors)
        
    # Sort by score descending
    ranked = sorted(
        zip(candidate_movie_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked

# Full pipeline: retrieve → rank
candidates, _ = retrieve_candidates(user_id=1, top_k=100)
ranked = rank_candidates(user_id=1, candidate_movie_ids=candidates)

print(f"\nFinal ranked recommendations for User 1:")
for movie_id, score in ranked[:5]:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"  {title} (score: {score:.4f})")    