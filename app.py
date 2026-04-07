import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================

# Note: Ensure 'movies.csv' and 'ratings.csv' (from MovieLens dataset) are in your directory
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    print("Error: Please ensure 'movies.csv' and 'ratings.csv' are in the project folder.")
    exit()

# Merge datasets for collaborative filtering tasks later
movie_ratings = pd.merge(movies, ratings, on='movieId')
movie_ratings.dropna(inplace=True)

print("Data Loaded Successfully.")
print(movies.head())

# ==========================================
# 2. CONTENT-BASED FILTERING (By Genre)
# ==========================================

# Initialize TF-IDF Vectorizer to turn genres into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute Cosine Similarity between all movies based on genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to their indices
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies_by_content(title, cosine_sim=cosine_sim):
    if title not in movie_indices:
        return "Movie title not found in database."
    
    # Get index of the movie
    idx = movie_indices[title]
    
    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of the top 10 most similar movies (excluding itself)
    sim_scores = sim_scores[1:11]
    movie_indices_similar = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices_similar]

# ==========================================
# 3. COLLABORATIVE FILTERING (By User Taste)
# ==========================================

# Setup Surprise reader and load the dataframe
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)

# Split into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use SVD (Singular Value Decomposition) algorithm
algo = SVD()
algo.fit(trainset)

# Evaluate the model
predictions = algo.test(testset)
print(f"Collaborative Filtering Model RMSE: {rmse(predictions):.4f}")

def recommend_for_user(user_id, n_recommendations=10):
    # Get list of all unique movie IDs
    all_movie_ids = movie_ratings['movieId'].unique()
    
    # Predict ratings for movies the user hasn't seen (or all movies)
    user_predictions = [algo.predict(user_id, m_id) for m_id in all_movie_ids]
    
    # Sort predictions by estimated rating (highest to lowest)
    user_predictions = sorted(user_predictions, key=lambda x: x.est, reverse=True)
    
    # Get top N movie IDs
    top_n_ids = [pred.iid for pred in user_predictions[:n_recommendations]]
    
    return movies[movies['movieId'].isin(top_n_ids)]['title']

# ==========================================
# 4. TESTING THE SYSTEM
# ==========================================

print("\n--- Content-Based Recommendations for 'Toy Story (1995)' ---")
print(recommend_movies_by_content('Toy Story (1995)'))

print("\n--- Collaborative Recommendations for User ID: 1 ---")
print(recommend_for_user(user_id=1))
